# CPT with LoRA and Layer Freezing

This notebook demonstrates how to pretrain LLaMA models using Layer Freezing and LoRA (Low-Rank Adaptation) techniques with the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. It includes data preprocessing, training, inference, and model merging steps.

## Features

- **Layer Freezing:** Efficiently CPT by freezing 4 layers.
- **LoRA:** Use parameter-efficient CPT with LoRA adapters.
- **Custom Dataset:** Preprocess Wikipedia and Markdown files for training.
- **Training & Inference:** Scripts for both training and interactive inference.
- **Model Merging:** Merge LoRA adapters or frozen layers into the base model.

## Setup

1. **Clone the repo:**
    ```bash
    git clone --depth "repo-name"
    Run "cell by cell" 
    ```

2. **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Install Jupyter Notebook:**
    ```bash
    pip install notebook
    ```

## Data Preparation

- **Wikipedia Dataset:**  
  Downloads and preprocesses a subset (e.g., 5%) of the English Wikipedia using Hugging Face `datasets`.

- **Markdown Files:**  
  Collects and processes Markdown documents from a specified directory.

- **Dataset Info:**  
  Updates `dataset_info.json` to register new datasets for LLaMA-Factory.

## Training

- **Layer Freezing:**  
  Configure and run training with frozen layers using `llamafactory-cli`.

- **LoRA:**  
  Configure and run LoRA-based training. Supports checkpoint resuming.

## Inference

- Use the trained model for interactive chat or batch inference.
- Example command:
    ```bash
    llamafactory-cli chat inference_config.json
    ```

## Model Merging

- Merge LoRA adapters or frozen layers into the base model for deployment or sharing.

## Example Usage

See the notebook `CPT_Lora_and_layerFreezing.ipynb` for step-by-step code and explanations.

## Requirements

- Python 3.8+
- torch
- bitsandbytes
- datasets
- huggingface_hub
- ipywidgets
- llamafactory