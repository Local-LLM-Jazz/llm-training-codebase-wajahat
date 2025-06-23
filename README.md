# CPT LLaMA Factory: 1% Pretraining & Instruction Tuning with Layer Freezing and LoRA

## Overview

This repository demonstrates how to pretrain and instruction-tune a LLaMA-based model on 1% of the Wikipedia dataset using:

* **CPT (Continual PreTraining)** on a small subset (1%) of data
* **Instruction Tuning** with:

  * Layer Freezing
  * **LoRA (Low-Rank Adaptation)** adapters

By combining these techniques, you can efficiently adapt large language models even under resource constraints.

## Table of Contents

1. [Installation](#installation)
2. [Dataset & Preprocessing](#dataset--preprocessing)
3. [Continual Pretraining (CPT)](#continual-pretraining-cpt)
4. [Instruction Tuning](#instruction-tuning)

   * [Layer Freezing](#layer-freezing)
   * [LoRA Adapter](#lora-adapter)
5. [Inference](#inference)
6. [Progress Report](#progress-report)
7. [Next Steps](#next-steps)

## Installation

1. Clone this repository and enter the project directory:

   ```bash
   git clone https://github.com/Local-LLM-Jazz/llm-training-codebase-wajahat.git
   cd llm-training-codebase-wajahat
   ```
2. Clone and **Run** with GPU support:
3. Install remaining dependencies:

   ```bash
   pip3 install datasets
   pip3 install -r requirements.txt
   ```
4. Verify GPU availability:

   ```python
   import torch
   print("CUDA version:", torch.version.cuda)
   print("CUDA available:", torch.cuda.is_available())
   print("Device name:", torch.cuda.get_device_name(0))
   ```
5. Authenticate with Hugging Face (if you plan to push or pull models):

   ```bash
   pip3 install huggingface_hub
   ```

   Then, in a Python shell or notebook:

   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

## Dataset & Preprocessing

* Uses a subset of Wikipedia (1%) stored as a JSON file (`wiki_1percent.json`)
* Expected format:

  ```json
  {
    "wiki_1percent": {
      "file_name": "wiki_1percent.json",
      "columns": { "prompt": "text" }
    }
  }
  ```
* Preprocessing steps:

  1. Load and tokenize text
  2. Filter and format into prompt–response pairs
  3. Save processed data for CPT and tuning stages

## Continual Pretraining (CPT)

1. Define the CPT data loader on the 1% dataset
2. Resume training from the base LLaMA checkpoint
3. Train for a small number of steps to adapt to domain corpus

## Instruction Tuning

After CPT, perform two types of instruction tuning:

### Layer Freezing

* Freeze lower layers of the transformer
* Train only the top \$k\$ layers on the instruction dataset

### LoRA Adapter

* Insert LoRA modules into query and value projections
* Train adapter weights (low-rank updates) while keeping base model frozen

## Inference

* Load your fine-tuned checkpoint
* Run the provided inference scripts to generate responses:

  ```python
  from llama_factory.infer import InferenceEngine

  engine = InferenceEngine(
      model_path="./checkpoints/cpt_1pct_instruction_lora",
      tokenizer_path="./tokenizer"
  )
  output = engine.generate("Translate English to French: Hello world.")
  print(output)
  ```
  ---
