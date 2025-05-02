# Medical LLM DPO Training

This project implements a complete pipeline for training a medical language model using DPO and PPO on PubMedQA dataset.

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
python dpo_train.py
```

3. Fine-tune the DPO-trained model using PPO
```bash
python train.py
``` 

3. Quantize and test the model:
```bash
python quantize.py
```

## Project Structure

- `data_generation.py`: Generates RLAIF training data using OpenBioLLM
- `dpo_train.py`: Runs preference optimization via DPO
- `train.py`: Reinforcement Learning with AI Feedback (PPO fine-tuning)
- `quantize.py`: Handles model quantization and testing
- `requirements.txt`: Project dependencies

## Notes

- The training process is optimized for a single A100 GPU
- Model quantization is performed using 4-bit precision
- Training data is generated from PubMedQA dataset
- The base model used is `Llama-medx_v3.2`
