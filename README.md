# GGUF Quantization Docs (Unofficial)

## Table of Contents

### Explainers
- [Legacy Quants](legacy-quants.md)
- [K-Quants](k-quants.md)
- [I-Quants](i-quants.md)
- [Importance Matrix](importance-matrix.md)
- [Naming Convention](naming.md)

### Practical Guides
- [Commands](commands.md)
- [Benchmarks](benchmarks.md)

## What is GGUF quantization?
*GGUF quantization* is an umbrella term for an LLM quantization ecosystem that includes:
- [GGML](https://github.com/ggml-org/ggml) (tensor library for machine learning);
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (LLM inference engine mostly targeting CPUs);
- [GGUF](https://huggingface.co/docs/hub/en/gguf) (binary file format for storing quantized models).

GGUF quantization implements *Post-Training Quantization* (PTQ): given an already-trained Llama-like model in high precision, it reduces the bit width of each individual weight. The resulting checkpoint requires less memory and thus facillitates inference on consumer-grade hardware.

## Who built it? Why are there no official docs?
GGUF was inspired by previous PTQ methods, including [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [QLoRA](https://arxiv.org/abs/2305.14314) and [QuIP#](https://arxiv.org/abs/2402.04396). But unlike most prior work that came out of research labs, the GGUF ecosystem was developed by the prolific open-source contributor [Georgi Gerganov](https://github.com/ggerganov) and a few others.

Writing docs and papers is simply not their priority, see [this comment](https://github.com/ggml-org/llama.cpp/pull/1684#issuecomment-2474462323):

<img src="images/no-papers.png" alt="No Papers" style="max-height: 200px;">

As the ecosystem rapidly grew over time, people are confused about the various algorithm iterations and settings.

## What is this repository?

This repository serves as unofficial documentation for the GGUF quantization ecosystem.

It written mostly manually by a [human](https://x.com/juliarturc). Any sections written by AI will be clearly flagged as ⚠️🤖.

## Contributing

Contributions are more than welcome! If you find mistakes or omissions, feel free to submit a pull request.

Just a few simple rules:
- **Reliable references**: PRs should be supported by reliable references (e.g. code and author comments from the official [llama.cpp](https://github.com/ggml-org/llama.cpp) repository). Medium articles and Reddit threads don't qualify.
- **No AI slop**: We all know when something is written by AI. Please only contribute when you have a human urge for expression.
