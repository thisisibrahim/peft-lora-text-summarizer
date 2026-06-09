# PEFT LoRA Text Summarizer

This repository is a hands-on learning project focused on fine-tuning a Transformer-based sequence-to-sequence model for dialogue summarization using **PEFT** and **LoRA**.

The goal of this project was not just to run a copied fine-tuning script, but to understand the complete workflow behind parameter-efficient fine-tuning: loading a base model, preparing a dataset, applying LoRA adapters, training only a small percentage of model parameters, saving the adapter, loading it back for inference, and evaluating the results against the base model.

---

## Project Overview

This project fine-tunes a summarization model on the **SAMSum** dialogue summarization dataset.

The model takes a conversation as input and generates a short summary.

Example input:

```text
John: Hey Sara, are you joining the project meeting today?
Sara: I might be late. I have a doctor's appointment.
John: No problem. Should I update the team?
Sara: Yes, please tell them I will join after 30 minutes.
John: Sure, I will inform everyone.
```

Expected summary:

```text
Sara will join the project meeting 30 minutes late because of a doctor's appointment, and John will inform the team.
```

---

## What This Project Demonstrates

This repository covers the complete PEFT/LoRA workflow:

* Loading a pre-trained sequence-to-sequence Transformer model
* Loading and inspecting the SAMSum dialogue summarization dataset
* Testing baseline model inference before fine-tuning
* Inspecting model layers to identify valid LoRA target modules
* Applying LoRA adapters using Hugging Face PEFT
* Training only a small fraction of the model parameters
* Saving the trained LoRA adapter separately from the base model
* Loading the base model with the trained adapter for inference
* Comparing base model and LoRA model outputs manually
* Evaluating both models using ROUGE scores

---

## Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face PEFT
* Hugging Face Evaluate
* ROUGE Score
* SentencePiece

---

## Model and Dataset

### Base Model

The project was tested with:

```text
google/flan-t5-small
```

The model can be changed from `config.py`.

### Dataset

The project uses:

```text
knkarthick/samsum
```

SAMSum is a dialogue summarization dataset containing messenger-style conversations and human-written summaries.

---

## Why PEFT and LoRA?

Full fine-tuning updates all model parameters, which can be expensive and resource-heavy.

PEFT, or Parameter-Efficient Fine-Tuning, allows us to fine-tune only a small number of additional parameters while keeping the original model mostly frozen.

LoRA, or Low-Rank Adaptation, adds small trainable matrices to selected layers of the model. In this project, LoRA adapters are applied to the attention projection modules of the model.

During training, only a small percentage of the total parameters are updated.

Example from the training run:

```text
trainable params: 344,064 || all params: 77,305,216 || trainable%: 0.4451
```

This means the project trained less than 1% of the full model parameters while keeping the base model intact.

---

## Project Structure

```text
peft-lora-text-summarizer/
│
├── config.py              # Project configuration
├── train.py               # PEFT/LoRA training script
├── infer.py               # Inference using trained LoRA adapter
├── evaluate_model.py      # Manual comparison: base model vs LoRA model
├── compare_rouge.py       # ROUGE comparison between base and LoRA model
├── baseline_infer.py      # Baseline inference using the base model
├── check_dataset.py       # Dataset inspection script
├── inspect_model.py       # Model module inspection for LoRA targets
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── outputs/               # Local model outputs and adapter files
└── samples/               # Optional sample inputs
```

Note: `outputs/` is ignored by Git because it may contain local checkpoints and adapter weights.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/peft-lora-text-summarizer.git
cd peft-lora-text-summarizer
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

All major settings are stored in `config.py`.

Example:

```python
MODEL_NAME = "google/flan-t5-small"
DATASET_NAME = "knkarthick/samsum"

OUTPUT_DIR = "./outputs/run_5000"
ADAPTER_DIR = "./outputs/lora_adapter_5000"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

TRAIN_SAMPLE_SIZE = 5000
EVAL_SAMPLE_SIZE = 300

BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
```

This makes it easy to experiment with different models, dataset sizes, learning rates, batch sizes, and adapter output folders.

---

## Step-by-Step Workflow

### 1. Check Dataset

```bash
python check_dataset.py
```

This verifies that the dataset loads correctly and contains the required columns:

```text
dialogue
summary
```

---

### 2. Run Baseline Inference

```bash
python baseline_infer.py
```

This checks how the base model performs before LoRA fine-tuning.

---

### 3. Inspect Model Modules

```bash
python inspect_model.py
```

This helps identify valid LoRA target modules.

For T5-style models, the project uses:

```python
target_modules=["q", "v"]
```

---

### 4. Train LoRA Adapter

```bash
python train.py
```

This script:

* loads the dataset
* loads the base model
* applies LoRA adapters
* tokenizes dialogue-summary pairs
* trains only the adapter parameters
* saves the trained adapter

---

### 5. Run Inference with LoRA Adapter

```bash
python infer.py
```

This loads the base model and attaches the trained LoRA adapter for summarization.

---

### 6. Compare Base Model and LoRA Model

```bash
python evaluate_model.py
```

This prints sample dialogues, reference summaries, base model summaries, and LoRA fine-tuned summaries.

---

### 7. Compare ROUGE Scores

```bash
python compare_rouge.py
```

This compares the base model and the LoRA fine-tuned model using ROUGE metrics.

---

## Experimental Findings

The project successfully demonstrated a complete PEFT/LoRA fine-tuning workflow.

One important learning from the experiments was that fine-tuning does not automatically guarantee better performance. In some runs, the LoRA adapter trained successfully but did not consistently outperform the already instruction-tuned base model.

This made the project valuable from an experimentation perspective because it showed the importance of:

* choosing the right base model
* using enough training data
* selecting suitable LoRA configuration
* tuning learning rate and epochs carefully
* comparing against a baseline model
* using both manual inspection and quantitative evaluation
* understanding that lower training loss does not always mean better generation quality

---

## Key Learning Outcomes

Through this project, I learned:

* how PEFT differs from full fine-tuning
* how LoRA adapters are attached to Transformer layers
* how to prepare dialogue-summary data for sequence-to-sequence training
* how to train only adapter parameters instead of the full model
* how to save and reload LoRA adapters
* how to compare model outputs before and after fine-tuning
* how to evaluate summarization models using ROUGE
* why experiment design matters in machine learning

---

## Important Note

This project is primarily a learning-focused implementation. The objective was to understand the PEFT/LoRA workflow end-to-end rather than produce a production-grade summarization model.

Future improvements may include:

* training on the full SAMSum dataset
* experimenting with `t5-small` and `flan-t5-base`
* using a GPU for faster and deeper training
* testing different LoRA ranks and target modules
* increasing evaluation sample size
* adding experiment tracking
* publishing trained adapter weights separately

---

## Status

Completed as a learning project.

The repository contains a working PEFT/LoRA fine-tuning pipeline and can be used as a foundation for future experiments in text summarization, instruction tuning, and parameter-efficient model adaptation.
