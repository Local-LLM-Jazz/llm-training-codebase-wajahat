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

See the notebook `CPT_Lora_and_layerFreezing.ipynb` for step-by-step code and explanations


# LLaMA-Factory: Pretraining & Instruction Tuning with Layer Freezing and LoRA([Clone my repo & run this notebook](https://github.com/Local-LLM-Jazz/llm-training-codebase-wajahat/blob/main/CPT_LLaMA_Factory_1Per_Pretraining_Instruction_tuning_withLayerFreezingandLora.ipynb))

This notebook demonstrates how to use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for continual pretraining and instruction tuning with **Layer Freezing** using LLaMA models  and apply instruction tuning **LoRA** techniques on CPT checkpoints. The workflow includes data preparation, model training, inference, merging, and evaluation.

---


## Features

- **Layer Freezing:** Efficiently CPT & finetune large models by freezing most layers.
- **LoRA:** Parameter-efficient finetuning using LoRA adapters.
- **Custom Dataset Support:** Preprocesses Wikipedia, Markdown, and Alpaca datasets.
- **Training & Inference Scripts:** End-to-end workflow for model development.
- **Model Merging:** Merge adapters or frozen layers into the base model for deployment.
- **Evaluation:** Includes scripts for general capability and NLG quality evaluation.

---

## Data Preparation

- **Wikipedia Data:**  
  Download and preprocess a 1% subset of English Wikipedia using Hugging Face `datasets`.

- **Markdown Data:**  
  Collect and preprocess Markdown files from your specified directory.

- **Alpaca Data:**  
  Download and format the Alpaca dataset for instruction tuning.

- **Dataset Registration:**  
  Update `dataset_info.json` to register new datasets for LLaMA-Factory.

---

## Pretraining with Layer Freezing

1. **Configure training** in `CPT_LayerFreezing_1per.json` (example config provided in the notebook).
2. **Run pretraining:**
    ```bash
    llamafactory-cli train CPT_LayerFreezing_1per.json
    ```

---

## Instruction Tuning

### Instruction Tuning with Layer Freezing

1. **Configure instruction tuning** in `InstructionTuning_LayerFreezing_Alpaca.json`.
2. **Run instruction tuning:**
    ```bash
    llamafactory-cli train InstructionTuning_LayerFreezing_Alpaca.json
    ```

### Instruction Tuning with LoRA

1. **Configure LoRA instruction tuning** in `InstructionTuning_LoRA_Alpaca.json`.
2. **Run LoRA instruction tuning:**
    ```bash
    llamafactory-cli train InstructionTuning_LoRA_Alpaca.json
    ```

---

## Inference

- Use the trained or merged model for interactive chat:
    ```bash
    llamafactory-cli chat <your_inference_config>.json
    ```
- Example code for interactive CLI chat is provided in the notebook.

---

## Model Merging

- Merge frozen or LoRA-tuned weights into the base model for deployment:
    ```bash
    llamafactory-cli export export_freeze_model.json
    ```

---

## Evaluation

- **General Capability (e.g., MMLU):**
    ```bash
    llamafactory-cli eval <your_eval_config>.json
    ```

- Evaluation configs for both pretraining and instruction tuning are provided in the notebook.

---

## Requirements

- Python 3.8+
- torch
- bitsandbytes
- datasets
- huggingface_hub
- ipywidgets
- llamafactory