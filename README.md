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
- __Dataset Info:__  
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
- __Dataset Registration:__  
   Update `dataset_info.json` to register new datasets for LLaMA-Factory.

---

## Pretraining with Layer Freezing

1. __Configure training__ in `CPT_LayerFreezing_1per.json` (example config provided in the notebook).

2. **Run pretraining:**

```bash
llamafactory-cli train CPT_LayerFreezing_1per.json
```

---

## Instruction Tuning

### Instruction Tuning with Layer Freezing

1. __Configure instruction tuning__ in `InstructionTuning_LayerFreezing_Alpaca.json`.

2. **Run instruction tuning:**

```bash
llamafactory-cli train InstructionTuning_LayerFreezing_Alpaca.json
```

### Instruction Tuning with LoRA

1. __Configure LoRA instruction tuning__ in `InstructionTuning_LoRA_Alpaca.json`.

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

# Experiment Results Summary

This section provides a summary of training experiments performed using the `20231101.en` Wikipedia dataset (and the Alpaca dataset for one run). All experiments were trained for a single epoch. Below is a breakdown of each method, dataset, and performance on MMLU.

## Table of Results

| Method                        | Data % | Epochs | Dataset     | Train Loss | MMLU Avg |  STEM | Social Sciences | Humanities | Other |
| ----------------------------- | :----: | :----: | :---------- | :--------: | :------: | :---: | :-------------: | :--------: | :---: |
| **CPT\_Layer\_Freezing**      |   5%   |    1   | 20231101.en |   2.1608   |   59.26  | 49.14 |      68.51      |    56.07   | 64.53 |
| **CPT\_LoRA**                 |   5%   |    1   | 20231101.en |   0.0715   |   56.91  | 47.71 |      66.92      |    51.80   | 63.42 |
| **CPT\_Layer\_Freezing**      |   1%   |    1   | 20231101.en |   2.2117   |   59.68  | 49.87 |      68.57      |    56.30   | 65.27 |
| **instructionTuning\_Freeze** |   1%   |    1   | 20231101.en |   1.4095   |   58.80  | 47.98 |      67.99      |    55.71   | 64.62 |
| **instructionTuning\_LoRA**   |   1%   |    1   | Alpaca      |   1.2925   |   58.41  | 49.30 |      67.21      |    54.79   | 63.79 |

## Overview

* **CPT (Continual Pretraining Tasks)**

  * *Layer Freezing*: Freeze last 4 layers.
  * *LoRA*: Apply Low-Rank Adaptation modules.

* **Instruction Tuning**

  * Fine-tuning on instruction data built on top of pretraining checkpoints.
  * *Freeze*: Instruction tuning while freezing pretrained layers.
  * *LoRA*: Instruction tuning using LoRA modules.

## Datasets

* **20231101.en**: Wikipedia dataset (Hugging Face `wikimedia/wikipedia` viewer snapshots).
* **Alpaca**: Instruction-following dataset used only for the `instructionTuning_LoRA` run.

## Training Setup

* **Epochs**: 1
* **Optimizer**: (AdamW)
* **Learning rate**: (0.0001)
* **Hardware**: (e.g., NVIDIA V100, A100)

---

## Requirements

- Python 3.8+
- torch
- bitsandbytes
- datasets
- huggingface_hub
- ipywidgets
- llamafactory