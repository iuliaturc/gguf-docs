# Importance Matrix

## Overview

The importance matrix (also called *"imatrix"*) is a quality enhancement that can be applied to any of the quants. They were introduced at the same time as [I-quants](i-quants.md). While the low-bit I-quants (e.g. IQ2) do require an imatrix for decent quality, the imatrix remains an orthogonal concept, and can be equally applied to [legacy](legacy-quants.md) and [K-quants](k-quants.md).

The core insight is that **not all model weights are equally important**. A weight is *important* if changing it by a small amount causes disproportionately large changes in the model output. Such weights should be allocated more precision. As we'll see soon, "allocating more precision" doesn't mean allocating more bits, but rather choosing the quantization constants (the scale `S` and zero-point `Z` if present) in a way that favors the important weights.

## What is the importance matrix?
For a weight matrix `W`, the importance matrix `I` has the same dimensions as `W` and assigns a real-valued score to each individual weight.

## How is the importance matrix computed?

<img src="images/importance-matrix.png" alt="importance-matrix" style="max-height: 400px;">

### The calibration set
Since the definition of weight importance is tied to model behavior ("important weights cause big changes in model output"), we need to watch the model behavior on a *calibration set*. It's typical to use a few hundred instances from [Wikitext](https://huggingface.co/datasets/Salesforce/wikitext), a subsest of Wikipedia. There have been some discussions about [overfitting](https://www.reddit.com/r/LocalLLaMA/comments/1993iro/ggufs_quants_can_punch_above_their_weights_now/), though the authors claim overfitting is unlikely (see [comment](https://github.com/ggml-org/llama.cpp/discussions/5006#discussioncomment-8166807)).

Given a calibration dataset, the importance matrix is calculated based on the *activation* values, an idea borrowed from the [AWQ](https://arxiv.org/abs/2306.00978) paper.

### The math
Consider a weight matrix `W` with dimensions `NxM`. Running inference on a calibration datapoint, we get `Wx = y` (`x` = input activation `Mx1`, `y` = output activation `Nx1`). So the output activation vector `y` has one entry for each row in `W`. This gives us **per-row importance scores**:

```
# per-row importance score
I_i = y_i^2
```

Next, we can compute the full importance matrix `I` by combining the activation-based per-row importance score `I_i` with the magnitude of the corresponding weight:
```
# per-weight importance score
I_ij = I_i + sqrt(σ^2 + w_ij^2) = y_i^2 + sqrt(σ^2 + w_ij^2)
```
σ is the weight variation across the row (it's probably there to make sure we get a non-zero value when `w_ij` is zero).

## How does the importance matrix influence quantization?

<img src="images/imatrix-objective.png" alt="imatrix-objective" style="max-height: 400px;">

As mentioned above, the importance matrix redirects precision towards important weights, but not as you would expect. It doesn't allocate more bits, but rather helps choosing quantization constants `S` and `Z` which will *favor* important weights: when an important weight is de-quantized during inference, its reconstruction error (delta between original and reconstructed value) will be lower than for other weights.

### De-coupling quantization and de-quantization constants
So far, the same set of constants `S` and `Z` was used both for quantization (float → integer) and de-quantization (integer → float). But now we want to *bias* the de-quantization process towards important weights. So we'll allocate a separate set of constants `S'` and `Z'` that we can manipulate for de-quantization.

### Finding optimal S' and Z'
GGUF treats this as a data-driven optimization problem. The ultimate goal is to minimize the reconstruction error across all entries `ij` of a matrix `W`:
```
L = \sum_i \sum_j (w_ij - w'_ij)^2
```
where `w'_ij` is the reconstructed weight:
```
q_i = quant(w_ij, S, Z)
w'_ij = dequant(q_i, S', Z') = dequant(quant(w_ij, S, Z), S', Z')
```
Rewriting the loss:
```
L = \sum_i \sum_j (w_ij - dequant(quant(w_ij, S, Z), S', Z'))^2
```
For fixed `S` and `Z` (see [legacy quants](legacy-quants.md) how to calculate them), this is a quadratic function with parameters `S'` and `Z'`, with a closed-form solution.

To be annoyingly pedantic, this loss is actually an *expectation over all possible inputs*. Just like in training, we approximate this expectation with a small dataset (we can reuse the calibration set from above).

### Grid search for S
To squeeze the last bit of performance, GGUF does a small grid search around the fixed value `S` (it considers 36 values in its vecinity, to be more exact). Adjusting the value of `S` translates to adjusting the clipping range: a larger value of `S` will clip the high-magnitude weights and re-allocate more precision to the ones closer to zero. So the final algorithm for finding (de-)quantization constants is:

<img src="images/imatrix-code.png" alt="imatrix-code" style="max-height: 400px;">

## What overhead does the importance marix incur?
None during inference! The importance matrix simply leads to more strategic choices of quantization constants. By inspecting a quantized checkpoint, there's virtually no way to tell whether the imporance matrix was used, since it doesn't store any extra values. (This is frustrating for some people, since they can't trace back how a checkpoint was quantized).

There's just a bit more pre-processing work to (a) pick a calibration dataset, (b) compute the imatrix, and (c) pass it to the quantization binary.

## Performance Impact
⚠️ Section written by Claude Code, proceed with caution.

### Quality Improvements
- **Significant gains**: 10-30% perplexity improvement common with importance matrix
- **Larger at low bit rates**: More dramatic improvements for aggressive quantization
- **Model-dependent**: Larger models often see bigger relative improvements

### Computational Overhead
- **Generation time**: Zero additional cost during inference
- **Quantization time**: Modest increase (10-20%) during model conversion
- **Storage**: Negligible increase in final model size

### Memory Requirements
- **Calibration**: Requires loading full model during importance matrix generation
- **Quantization**: Standard memory requirements for quantization process
- **Inference**: No additional memory overhead

## Best Practices
⚠️ Section written by Claude Code, proceed with caution.

### Calibration Dataset Selection
- **Use representative data**: Match your intended use case as closely as possible
- **Quality over quantity**: 200-500 high-quality examples often sufficient
- **Avoid overfitting**: Don't use your test/evaluation data for calibration
- **Consider domain**: Legal, medical, code, and other domains benefit from domain-specific calibration

### Quantization Strategy
- **Always use with aggressive quantization**: Importance matrix most beneficial for `Q4` and below
- **Combine with appropriate base method**: [I-quants](i-quants.md) for extreme compression, [K-quants](k-quants.md) for balanced approach
- **Validate results**: Always evaluate quantized model performance on held-out data

### Workflow Integration
1. **Generate importance matrix** using representative calibration data
2. **Quantize model** with importance matrix for optimal quality
3. **Evaluate performance** on target tasks
4. **Iterate if needed** with different calibration data or quantization settings

### When NOT to Use Importance Matrix
- **High precision quantization**: Minimal benefit for `Q8` or higher precision
- **Resource constraints**: Skip if calibration dataset unavailable or computational resources limited
- **Simple deployments**: May not justify additional complexity for some use cases

---
[← Back: I-Quants](i-quants.md) | [Next: Practical Guide →](practical-guide.md)