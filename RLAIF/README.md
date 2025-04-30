# Medical LLM DPO Training

This project implements a complete pipeline for training a medical language model using Direct Preference Optimization (DPO) on PubMedQA dataset.

## Requirements

- Python 3.8+
- CUDA-capable GPU (A100 recommended)
- 16GB+ GPU memory

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Generate RLAIF training data:
```bash
python data_generation.py
```

2. Train the model using DPO:
```bash
python train.py
```

3. Quantize and test the model:
```bash
python quantize.py
```

## Project Structure

- `data_generation.py`: Generates RLAIF training data using OpenBioLLM
- `train.py`: Implements DPO training using TRL
- `quantize.py`: Handles model quantization and testing
- `requirements.txt`: Project dependencies

## Notes

- The training process is optimized for a single A100 GPU
- Model quantization is performed using 4-bit precision
- Training data is generated from PubMedQA dataset
- The base model used is Llama-medx_v3.1 