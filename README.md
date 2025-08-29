# QLoRA Fine-Tuning of Phi-2

This repository contains an experiment fine-tuning **[microsoft/phi-2](https://huggingface.co/microsoft/phi-2)** using **QLoRA** (Quantized Low-Rank Adaptation). The fine-tuning is performed on the **OpenAssistant dataset**, targeting instruction-following alignment.

---

## 📌 Project Overview

* **Base Model**: Phi-2 (2.7B parameters)
* **Fine-Tuning Method**: QLoRA with 4-bit quantization (bnb.int4)
* **Frameworks**: Hugging Face `transformers`, `trl`, `peft`
* **Dataset**: OpenAssistant conversations (instruction/response pairs)
* **Compute**: NVIDIA T4 (suitable for lightweight experimentation)

---

## ⚙️ Training Configuration

* **Sequence Length**: 256 tokens
* **Training Examples**: Capped at 32,000 for ~3–5 hour runs
* **Preview Prompt**: Generated every 100 optimizer steps to monitor progress
* **Quantization**: 4-bit loading of base model + LoRA adapters
* **Optimizer**: AdamW (via Hugging Face `trl` defaults)
* **Output Directory**: `phi2-oasst1-qlora`

---


## 🚀 Training Pipeline

1. **Load Dataset** – Preprocess JSONL input with text-only entries.
2. **Tokenizer Setup** – Phi-2 tokenizer, extended with special tokens.
3. **Model Load** – Quantized Phi-2 (bnb int4).
4. **Apply LoRA** – Inject LoRA adapters on attention/MLP layers.
5. **Train with TRL’s `SFTTrainer`** – Monitor loss and generate periodic previews.
6. **Save Checkpoints** – Adapters stored incrementally for resumption.

---

## 🔍 Example Behavior

**Before Fine-Tuning (vanilla Phi-2):**
```text
Instruct: Turn the following note into a concise, professional email (≤100 words) that
1) accepts responsibility for a missed deadline,
2) proposes a new deadline (Friday),
3) lists 3 quick next steps as bullet points,
4) ends with a courteous sign-off.

Note: "Sorry, I didn't finish the report. Can we push it to Friday?"
Output: Dear [Name],

I apologize for missing the deadline for the report. I understand that this may have caused inconvenience and I take full responsibility for it.

To rectify the situation, I propose that we push the deadline to Friday. This will allow us to complete the report and ensure that it meets the required standards.

In the meantime, I would like to outline three quick next steps:
1. Review the existing data and gather any additional information that may be needed.
2. Analyze the data and identify key insights.
3. Prepare a draft of the report and incorporate the findings.

I appreciate your understanding and cooperation. If you have any questions or require further assistance, please do not hesitate to reach out.

Best regards,
[Your Name]
```

* Responses are generic, verbose, and not instruction-following.

**After Fine-Tuning (QLoRA on OASST):**
```
Instruct: Turn the following note into a concise, professional email (≤100 words) that
1) accepts responsibility for a missed deadline,
2) proposes a new deadline (Friday),
3) lists 3 quick next steps as bullet points,
4) ends with a courteous sign-off.

Note: "Sorry, I didn't finish the report. Can we push it to Friday?"
Output: Dear [Recipient],

I am writing to apologize for missing the deadline for the report. I understand that this is an important project and I take full responsibility for the delay.

To make up for the delay, I propose that we push the deadline to Friday. I have already taken the following steps to ensure that the report is completed on time:

1. I have reviewed the report and made the necessary revisions.
2. I have reached out to the necessary stakeholders to gather any additional information.
3. I have scheduled a meeting with the team to discuss the report and ensure that everyone is on the same page.

I understand that this delay may have caused some inconvenience, and I apologize for any inconvenience caused. I am committed to delivering a high-quality report and will do everything in my power to ensure that it is completed on time.

```

* Produces structured, polite, instruction-following completions.
* Tends toward verbose answers due to dataset style.
* Example: Email formatting task now yields multi-step structured responses.

---

## 📊 Key Observations

* **Logs confirm correct weight loading**: All checkpoints matched.
* **Fine-tuning effect**: Noticeable shift in style to instruction-following, but sometimes ignores strict word limits (dataset bias).
* **Preview generations**: Help monitor improvement in structure and tone.

---

## 🛠️ Future Work

* Add dataset samples emphasizing **conciseness** (≤100 words) to enforce brevity.
* Experiment with **DPO/GRPO** for reward-driven fine-tuning on “conciseness”.
* Evaluate outputs on custom held-out prompts.

---

## 📎 References

* [Phi-2 on Hugging Face](https://huggingface.co/microsoft/phi-2)
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
* [TRL (Transformers Reinforcement Learning)](https://github.com/huggingface/trl)
* [OpenAssistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)