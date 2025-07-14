# Commands

A typical workflow would involve the following steps:

0. For convenience, set up environment variables:

    ```
    ORIGINAL_MODEL_GGUF=/path/to/gguf/model
    ORIGINAL_MODEL=/path/to/your/model # if your original model is not already in GGUF format:
    QUANTIZED_MODEL=/path/to/quantized/model
    QUANTIZATION_STRATEGY=IQ2
    ```


1. Install [llama.cpp](https://github.com/ggml-org/llama.cpp) following instructions in their README.

2. Ensure your original model is in GGUF format. If not:

    a. Clone the [llama.cpp](https://github.com/ggml-org/llama.cpp) if you haven't already in step 1:

    ```
    git clone https://github.com/ggml-org/llama.cpp.git
    ```

    b. Create a Python virtual environment and install dependencies:

        python3 -m venv myvenv
        source myvenv/bin/activate
        pip install -r llama.cpp/requirements.txt

    c. Run the Hugging Face converter:

        python3 llama.cpp/convert_hf_to_gguf.py $ORIGINAL_MODEL --outfile $ORIGINAL_MODEL_GGUF

3. (Optional) Produce an importance matrix:

    a. Download your desired dataset (from e.g. Hugging Face) and make sure it's in text format.

        export CALIBRATION_DATASET=/path/to/text/file

    b. Compute the imatrix:
        
        llama-imatrix -m $ORIGINAL_MODEL_GGUF -f $CALIBRATION_DATASET

4. Quantize the original model:
    ```
    llama-quantize $ORIGINAL_MODEL_GGUF $QUANTIZED_MODEL $QUANTIZATION_STRATEGY
    ```
    Optionally, you can pass the imatrix from step 3 via `--imatrix` (it's a `.dat` file).

5. Run inference with the quantized model:
    ```
    llama-cli -m $QUANTIZED_MODEL -p "Your prompt here"
    ```

---
[← Back: Importance Matrix](importance-matrix.md) | [Next: Benchmarks →](benchmarks.md)