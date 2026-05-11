# Fine-tune Once, Serve Anywhere

This repository contains the notebooks and results for an NLP II final project
that compares fine-tuning and inference workflows for a small open-source LLM.
The project uses the same base model and dataset across several serving
frameworks so their behavior, latency, throughput, and memory use can be
compared under similar conditions.

## Project Overview

The experiments are built around:

- **Base model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Dataset:** `databricks/databricks-dolly-15k`
- **Fine-tuning methods:** Hugging Face Transformers + PEFT/LoRA and Unsloth
- **Inference backends:** Transformers, Unsloth, vLLM, and Ollama
- **Execution target:** Google Colab with a T4 GPU

The project is notebook-first: dependencies are installed inside each notebook,
and the notebooks are intended to be run independently in Colab.

## Repository Structure

```text
.
+-- README.md
+-- report.pdf
+-- results/
|   +-- inference_results.csv
|   +-- inference_1.png
|   +-- inference_2.png
|   +-- inference_3.png
|   +-- inference_4.png
|   +-- training_transformers.png
|   +-- training_unsloth.png
+-- src/
    +-- fine-tuning/
    |   +-- transformers.ipynb
    |   +-- unsloth.ipynb
    +-- inference/
        +-- transformers.ipynb
        +-- unsloth.ipynb
        +-- vLLM.ipynb
        +-- ollama.ipynb
```

## Notebooks

### Fine-tuning

`src/fine-tuning/transformers.ipynb`

Fine-tunes TinyLlama with Hugging Face Transformers, PEFT, LoRA, and
4-bit quantization through BitsAndBytes. It loads Dolly 15k, creates
stratified train/dev/test splits, tokenizes instruction-response examples,
trains a LoRA adapter, records memory usage, and evaluates the result with
ROUGE, BLEU, BERTScore, and sample human inspection prompts.

`src/fine-tuning/unsloth.ipynb`

Runs the same fine-tuning experiment using Unsloth's optimized model loading
and training flow. It uses TRL's `SFTTrainer`, the same dataset/model setup,
LoRA parameters, evaluation metrics, and training-curve visualization.

Shared fine-tuning configuration:

| Setting | Value |
| --- | --- |
| Seed | `42` |
| Dataset | `databricks/databricks-dolly-15k` |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Learning rate | `2e-4` |
| Epochs | `1` |
| LoRA rank | `16` |
| LoRA alpha | `32` |
| LoRA dropout | `0.05` |
| Max sequence length | `128` |
| Batch size target | `32` |

### Inference

`src/inference/transformers.ipynb`

Benchmarks direct Hugging Face Transformers inference on a shuffled sample of
Dolly 15k prompts.

`src/inference/unsloth.ipynb`

Benchmarks Unsloth inference for the same TinyLlama model and prompt sample.
This notebook requires GPU execution.

`src/inference/vLLM.ipynb`

Benchmarks vLLM serving with the same generation settings. It includes both a
75-prompt run and a single-prompt run. The notebook notes that vLLM manages GPU
memory differently from the other frameworks, so memory reporting is based on
the configured GPU utilization.

`src/inference/ollama.ipynb`

Installs and starts Ollama inside the notebook, pulls the `tinyllama` model,
sends requests through Ollama's local API, and measures latency, throughput,
and memory usage.

Shared inference configuration:

| Setting | Value |
| --- | --- |
| Dataset sample | 75 shuffled Dolly prompts, seed `42` |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Max new tokens | `128` |
| Temperature | `0.7` |
| Sampling | enabled |

## Results

The `results/` directory stores plots and the benchmark summary generated from
the notebooks.

Current inference benchmark summary:

| Framework | Avg latency (s) | Throughput (tok/s) | Memory (GB) |
| --- | ---: | ---: | ---: |
| Transformers | 2.34 | 38.88 | 2.07 |
| Unsloth | 2.58 | 36.86 | 2.07 |
| Ollama | 0.57 | 255.10 | 0.78 |
| vLLM (75 prompts) | 0.03 | 2785.53 | 8.84 |
| vLLM (1 prompt) | 1.49 | 85.86 | 8.84 |

Training plots:

- `results/training_transformers.png`
- `results/training_unsloth.png`

Inference plots:

- `results/inference_1.png`
- `results/inference_2.png`
- `results/inference_3.png`
- `results/inference_4.png`

The full written analysis is available in `report.pdf`.

## How to Run

The workflow is the same for all notebooks:

1. Open the notebook in Google Colab.
2. Select a GPU runtime: `Runtime > Change runtime type > T4 GPU`.
3. Run all cells from top to bottom.

Each notebook installs its own dependencies in the first cells. The inference
notebooks are independent from the fine-tuning notebooks unless you choose to
modify them to load a locally trained adapter.

## Notes

- GPU execution is recommended for every notebook and required for Unsloth,
  vLLM, and practical LoRA fine-tuning.
- Ollama is installed inside `src/inference/ollama.ipynb`, so that notebook
  expects a Linux-like Colab environment.
- Generated checkpoints, caches, logs, and virtual environments are excluded
  through `.gitignore`.
- The repository currently tracks notebooks and result artifacts, not reusable
  Python modules or a package entry point.
