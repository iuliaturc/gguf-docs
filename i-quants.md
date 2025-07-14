# I-Quants

## Overview

I-quants are the third generation of quantization algorithms in `llama.cpp`, marking a significant conceptual departure from both [legacy quants](legacy-quants.md) and [K-quants](k-quants.md). While previous methods used *uniform scalar* quantization, I-quants introduce *vector quantization* and the option to take *weight importance* into account when quantizing.

<img src="images/vector-quantization.png" alt="vector-quantization" style="max-height: 400px;">

## Scalar vs vector quantization

### Scalar quantization
Traditional quants ([legacy](legacy-quants.md) and [K-quants](k-quants.md)) quantize individual weights. This is *scalar* quantization:
```
w_i → integer_quant
```

### Vector quantization
In contrast, I-quants treat groups of 8 weights as indivisible. This is *vector* quantization:
```
w_vec = [w_1, w_2, ..., w_8] → integer_quant
```

Vector quantization is facillitated by a *codebook* of *reference* (or *prototype*) vectors, with entries of this form:
```
r_vec = [r_1, r_2, ..., r_8] → integer_code
```

## (Simplified) vector quantization algorithm
Conceptually, this is how a weight vector `w_vec` is quantized. In practice, there are additional optimizations (discussed in the next sections).

1. Find its nearest neighbor `r_vec` in the codebook
2. Calculate a scale `S = |w_vec| / |r_vec|`
3. Store:
  
    a. the *code* associated with `r_vec`, and 

    b. the scale `S`, which is shared across 256 vectors, similarly to the [Block Quantization](legacy-quants.md#block-quantization) method from legacy and K-quants.

While storing the scale allows us to recover the original magnitude of the weight vector, its exact orientation is forever lost. This means the codebook needs to cover as many angles of the 8-dimensional space as possible.

## Codebook design

<img src="images/codebook.png" alt="codebook" style="max-height: 400px;">

It's not clear how the reference vectors were chosen. They are simply hard-coded in [ggml-quants.h](https://github.com/ggml-org/llama.cpp/blob/0d9226763c82562186122f3b827fa3862864a19c/ggml/src/ggml-common.h#L482). In this file, every 8D codebook vector is encoded as single hexadecimal value, which obfuscates it from the casual reader. To make matters worse, the algorithm that decodes the hexadecimals into 8D vectors seems to differ based on the exact quant sub-type.

In [this PR](https://github.com/ggml-org/llama.cpp/pull/4773), Kawrakow cites the [QuIP#](https://arxiv.org/abs/2402.04396) paper as a source of inspiration. QuIP# borrows the reference vectors from the [E8 lattice](https://en.wikipedia.org/wiki/E8_lattice) (a collection of 240 8D vectors that are borderline mystical and were connected to the theory of everything!). However, there is no evidence in the code that they actually used the E8 lattice (likely, they cited QuIP# as an inspiration for vector quantization at large).

### Codebook optimization: the sign trick
This is a clever trick that keeps the codebook size small:
- Reference vectors are exclusively *positive*, in all dimensions.
- Instead of querying the codebook with `w_vec`, GGUF searches for the nearest neighbor of its absolute-value vector (i.e. flips the negative signs).
- Once the nearest `r_vec` is identified, the following are stored:

    a. the *code* associated with `r_vec` in the codebook (as above);

    b. the scale `S` (as above);

    c. the signs of the original `w_vec` dimensions, which can be neatly packed in a single byte, e.g. `0b01010101` means `w_vec` alternates positive and negative signs.

The additional sign byte increases storage requirements, but also increases the size of the codebook by a factor of `2^8 = 256`. Keeping the stored codebook small means less storage for the codebook and faster nearest neighbor search.

## Ultra-low bit rates
Even with the sign byte, I-quants achieve better compression rate than their predecessors. For instance, given:
- 8-dimensional vectors => **8 bits for signs**
- stored codebook size of 256 => **8-bit codes** (and *actual* codebook size = 256 * 256 = 65,536)
- FP16 scale `S` shared across 256 vectors (i.e. 2048 weights) => *0.008 bits for scale*

we get a compression rate of 2.008 bpw.

⚠️ This number might not be *entirely* accurate because GGUF uses more optimization tricks, especially for bit packing, but... give or take, this is the rough math behind `IQ2` quants.

## What does the I stand for?
The "I" in I-quants most likely stands for "importance," referencing the [importance matrix](importance-matrix.md) capability, though this feature can also be applied to [legacy quants](legacy-quants.md) and [K-quants](k-quants.md).

---
[← Back: K-Quants](k-quants.md) | [Next: Importance Matrix →](importance-matrix.md)