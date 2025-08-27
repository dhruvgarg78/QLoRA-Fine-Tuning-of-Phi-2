# QLoRA Fine-Tuning of Phi-2

This repository contains an experiment fine-tuning
**[microsoft/phi-2](https://huggingface.co/microsoft/phi-2)** using
**QLoRA** (Quantized Low-Rank Adaptation). The fine-tuning is performed
on the **OpenAssistant dataset**, targeting instruction-following
alignment.

------------------------------------------------------------------------

## 📌 Project Overview

-   **Base Model**: Phi-2 (2.7B parameters)\
-   **Fine-Tuning Method**: QLoRA with 4-bit quantization (bnb.int4)\
-   **Frameworks**: Hugging Face `transformers`, `trl`, `peft`\
-   **Dataset**: OpenAssistant conversations (instruction/response
    pairs)\
-   **Compute**: NVIDIA T4 (suitable for lightweight experimentation)

------------------------------------------------------------------------

## ⚙️ Training Configuration

-   **Sequence Length**: 256 tokens\
-   **Training Examples**: Capped at 32,000 for \~3--5 hour runs\
-   **Preview Prompt**: Generated every 100 optimizer steps to monitor
    progress\
-   **Quantization**: 4-bit loading of base model + LoRA adapters\
-   **Optimizer**: AdamW (via Hugging Face `trl` defaults)\
-   **Output Directory**: `phi2-oasst1-qlora`

------------------------------------------------------------------------

## 📂 Repository Structure

    Qlora_phi_2_fine_tune.ipynb    # Main notebook with training pipeline
    data/
      ├── train.jsonl              # Training data in {"text": "..."} format
      ├── val.jsonl                # Validation data (unused in current config)
    outputs/
      └── phi2-oasst1-qlora/       # Saved checkpoints and adapters

------------------------------------------------------------------------

## 🚀 Training Pipeline

1.  **Load Dataset** -- Preprocess JSONL input with text-only entries.\
2.  **Tokenizer Setup** -- Phi-2 tokenizer, extended with special
    tokens.\
3.  **Model Load** -- Quantized Phi-2 (bnb int4).\
4.  **Apply LoRA** -- Inject LoRA adapters on attention/MLP layers.\
5.  **Train with TRL's `SFTTrainer`** -- Monitor loss and generate
    periodic previews.\
6.  **Save Checkpoints** -- Adapters stored incrementally for
    resumption.

------------------------------------------------------------------------

## 🔍 Example Behavior

**Before Fine-Tuning (vanilla Phi-2):** - Responses are generic,
verbose, and not instruction-following.

**After Fine-Tuning (QLoRA on OASST):** - Produces structured, polite,
instruction-following completions. - Tends toward verbose answers due to
dataset style. - Example: Email formatting task now yields multi-step
structured responses.

------------------------------------------------------------------------

## 📊 Key Observations

-   **Logs confirm correct weight loading**: All checkpoints matched.\
-   **Fine-tuning effect**: Noticeable shift in style to
    instruction-following, but sometimes ignores strict word limits
    (dataset bias).\
-   **Preview generations**: Help monitor improvement in structure and
    tone.

------------------------------------------------------------------------

## 🛠️ Future Work

-   Add dataset samples emphasizing **conciseness** (≤100 words) to
    enforce brevity.\
-   Experiment with **DPO/GRPO** for reward-driven fine-tuning on
    "conciseness".\
-   Evaluate outputs on custom held-out prompts.

------------------------------------------------------------------------

## 📎 References

-   [Phi-2 on Hugging Face](https://huggingface.co/microsoft/phi-2)\
-   [QLoRA: Efficient Finetuning of Quantized
    LLMs](https://arxiv.org/abs/2305.14314)\
-   [TRL (Transformers Reinforcement
    Learning)](https://github.com/huggingface/trl)\
-   [OpenAssistant
    Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)
