# Nexuss AI: End-to-End Training, Fine-Tuning & Deployment Guide

**Author:** Tadiyos  
**Project:** Nexuss AI  
**Status:** Production Ready  
**Tokenizer:** Tiktoken (`cl100k_base`)  

---

## 📖 Table of Contents

1. [Introduction & Philosophy](#1-introduction--philosophy)
2. [Environment Setup](#2-environment-setup)
3. [Data Engineering Pipeline](#3-data-engineering-pipeline)
4. [Phase 1: Pre-Training (From Scratch)](#4-phase-1-pre-training-from-scratch)
5. [Phase 2: Supervised Fine-Tuning (SFT)](#5-phase-2-supervised-fine-tuning-sft)
6. [Phase 3: Advanced Techniques (LoRA, Freezing, QLoRA)](#6-phase-3-advanced-techniques-lora-freezing-qlora)
7. [Phase 4: Reinforcement Learning (RLHF/DPO)](#7-phase-4-reinforcement-learning-rlhfdpo)
8. [Catastrophic Forgetting & Knowledge Retention](#8-catastrophic-forgetting--knowledge-retention)
9. [Versioning, Checkpointing & Release Strategy](#9-versioning-checkpointing--release-strategy)
10. [Operational Workflows (Cookbook)](#10-operational-workflows-cookbook)
11. [Troubleshooting & Best Practices](#11-troubleshooting--best-practices)

---

## 1. Introduction & Philosophy

Nexuss AI is built on a **Pure From-Scratch** philosophy. We do not rely on inherited biases from pre-trained weights unless explicitly loaded for fine-tuning. This guide covers the entire lifecycle:
- **Pre-training**: Building the base intelligence.
- **Fine-Tuning**: Specializing for tasks.
- **Alignment**: Ensuring safety and helpfulness (RLHF).
- **Lifecycle Management**: Versioning, merging, and retaining knowledge.

### Core Capabilities
- ✅ **Pure Training**: Start from random initialization or specific architecture configs.
- ✅ **Flexible Fine-Tuning**: Full, LoRA, QLoRA, and Freeze strategies.
- ✅ **Knowledge Retention**: Strategies to prevent catastrophic forgetting.
- ✅ **Modular Releases**: Manage multiple adapters and merged models.

---

## 2. Environment Setup

### Prerequisites
- Python 3.8+
- GPU(s) with CUDA support
- `tiktoken` installed for tokenization

### Configuration
Ensure your `dataset_info.json` is configured in `/workspace/data/`.

**Tokenizer Setup:**
All training commands must include:
```bash
--use_tiktoken True --tiktoken_encoding cl100k_base
```

---

## 3. Data Engineering Pipeline

Data is the fuel. Poor data = Poor model.

### A. Pre-Training Data (Base Model)
**Goal:** Teach language structure, facts, and reasoning.
**Format:** Raw text (JSONL).
**File:** `pretrain_data.jsonl`

```json
{"text": "Nexuss AI is a next-generation large language model..."}
{"text": "Quantum computing leverages principles of quantum mechanics..."}
```
*Note: Clean, deduplicate, and filter low-quality text before training.*

### B. Instruction Fine-Tuning Data (SFT)
**Goal:** Teach the model to follow instructions.
**Format:** Conversational (Alpaca/ShareGPT style).
**File:** `sft_data.jsonl`

```json
{
  "messages": [
    {"role": "user", "content": "Explain quantum entanglement."},
    {"role": "assistant", "content": "Quantum entanglement is a phenomenon where..."}
  ]
}
```

### C. Reward Modeling / Preference Data (RLHF/DPO)
**Goal:** Align model with human values (Helpful, Honest, Harmless).
**Format:** Chosen vs. Rejected responses.
**File:** `preference_data.jsonl`

```json
{
  "prompt": "How do I make a cake?",
  "chosen": "Here is a delicious recipe...",
  "rejected": "I don't know, google it."
}
```

### Registration
Edit `data/dataset_info.json`:
```json
{
  "nexuss_pretrain": {
    "file_name": "pretrain_data.jsonl",
    "columns": { "prompt": "text" }
  },
  "nexuss_sft": {
    "file_name": "sft_data.jsonl",
    "formatting": "sharegpt"
  },
  "nexuss_dpo": {
    "file_name": "preference_data.jsonl",
    "formatting": "dpo"
  }
}
```

---

## 4. Phase 1: Pre-Training (From Scratch)

This phase builds the **Base Model**. It learns grammar, facts, and basic reasoning.

### Strategy
- **Architecture:** Define layers, heads, hidden size (e.g., Llama-3.2-1B architecture).
- **Objective:** Next Token Prediction.
- **Checkpointing:** Critical. Saves every N steps to resume if interrupted.

### Command
```bash
python src/train.py \
    --stage pt \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --use_tiktoken True --tiktoken_encoding cl100k_base \
    --do_train \
    --dataset nexuss_pretrain \
    --dataset_dir data \
    --template default \
    --finetuning_type full \
    --output_dir output/nexuss_base_v1 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --fp16 \
    --resume_from_checkpoint output/nexuss_base_v1/checkpoint-XXXX \
```
*Note: If starting purely from scratch without weights, ensure `model_name_or_path` points to a config-only directory or use a random init flag if supported by your specific backend.*

### Validation
- Monitor **Loss Curve**: Should decrease steadily.
- Perplexity Check: Run generation samples every 1000 steps.

---

## 5. Phase 2: Supervised Fine-Tuning (SFT)

Transform the Base Model into a Chat Assistant.

### Strategy
- **Input:** Base Model (`output/nexuss_base_v1`).
- **Data:** Instruction/Conversation pairs.
- **Method:** Full Fine-Tune (for maximum capability) or LoRA (for speed).

### Command (Full Fine-Tune)
```bash
python src/train.py \
    --stage sft \
    --model_name_or_path output/nexuss_base_v1 \
    --use_tiktoken True --tiktoken_encoding cl100k_base \
    --do_train \
    --dataset nexuss_sft \
    --dataset_dir data \
    --template default \
    --finetuning_type full \
    --output_dir output/nexuss_sft_full_v1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --fp16
```

---

## 6. Phase 3: Advanced Techniques (LoRA, Freezing, QLoRA)

Use these when resources are limited or you need modular task-specific models.

### A. LoRA (Low-Rank Adaptation)
Trains only small adapter matrices. Keeps base model frozen.
- **Pros:** Fast, low memory, swappable adapters.
- **Cons:** Slight performance drop vs full fine-tune.

```bash
python src/train.py \
    ... \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir output/nexuss_lora_taskA
```

### B. QLoRA (Quantized LoRA)
Loads base model in 4-bit precision. Ultra-low memory.
```bash
python src/train.py \
    ... \
    --finetuning_type lora \
    --quantization_bit 4 \
    --output_dir output/nexuss_qlora_taskA
```

### C. Freezing Layers
Freeze early layers (embeddings, lower transformers), train only top layers.
*Configuration handled via `freeze_params` in config or specific script flags.*

---

## 7. Phase 4: Reinforcement Learning (RLHF/DPO)

Align the model to be helpful and safe.

### Strategy
- **DPO (Direct Preference Optimization):** Stable, easier than PPO.
- **Input:** SFT Model + Preference Dataset.

### Command (DPO)
```bash
python src/train.py \
    --stage dpo \
    --model_name_or_path output/nexuss_sft_full_v1 \
    --use_tiktoken True --tiktoken_encoding cl100k_base \
    --do_train \
    --dataset nexuss_dpo \
    --dataset_dir data \
    --template default \
    --finetuning_type full \
    --output_dir output/nexuss_aligned_v1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-7 \
    --num_train_epochs 1
```

---

## 8. Catastrophic Forgetting & Knowledge Retention

**⚠️ CRITICAL WARNING:** The architecture does **NOT** automatically prevent forgetting. If you fine-tune on Task B, you **WILL** lose Task A capabilities unless you use these strategies:

### Strategy A: Data Mixing (Recommended)
Combine old and new data during training.
```json
// dataset_info.json
"mixed_dataset": {
  "file_name": ["old_task.jsonl", "new_task.jsonl"],
  "sampling_rate": [0.3, 0.7] 
}
```
*Keep 10-30% of old data in every new training run.*

### Strategy B: Separate Adapters (LoRA)
Do not merge adapters. Keep them separate.
- `base_model` + `adapter_task_A` → Run for Task A
- `base_model` + `adapter_task_B` → Run for Task B
*No forgetting occurs because the base weights never change.*

### Strategy C: Replay Buffer
When training Task B, explicitly sample batches from Task A's dataset.

### Strategy D: Reduced Learning Rate
When fine-tuning sequentially, reduce LR by 10x to minimize weight displacement.

---

## 9. Versioning, Checkpointing & Release Strategy

### Checkpointing (Auto-Save)
- Enabled via `--save_steps 500`.
- Saves optimizer state, scheduler, and weights.
- **Resume:** Use `--resume_from_checkpoint path/to/checkpoint-500`.

### Versioning Workflow
Directories act as versions:
- `output/nexuss_base_v1`
- `output/nexuss_sft_v1`
- `output/nexuss_sft_v2` (Improved data)

### Releasing a Model
#### Option 1: LoRA Adapter Release
Ship the small adapter file + Base Model reference.
*Size: ~100MB*

#### Option 2: Merged Standalone Release
Merge LoRA weights into the base model for a single deployable artifact.
```bash
python src/export.py \
    --model_name_or_path output/nexuss_base_v1 \
    --adapter_name_or_path output/nexuss_lora_taskA \
    --output_dir output/nexuss_merged_v1 \
    --merge_adapter true
```
*Size: Full model size (e.g., 2GB - 70GB)*

---

## 10. Operational Workflows (Cookbook)

### Scenario 1: Train Base → Fine-Tune → Release
1. **Pre-train:** `--stage pt` → Output: `base_v1`
2. **SFT:** `--stage sft --model base_v1` → Output: `sft_v1`
3. **Export:** Merge and save as `Nexuss-v1.0-Standalone`.

### Scenario 2: Add New Capability Without Forgetting
*Goal: Add Coding ability to a Chat model.*
1. **Prepare Data:** Mix 50% Chat Data + 50% Code Data.
2. **Train:** `--stage sft --dataset mixed_chat_code`.
3. **Result:** Model retains chat ability + gains coding.

### Scenario 3: Sequential Fine-Tuning (Risky but possible)
*Goal: Train Medical, then Legal.*
1. **Train Medical:** Create `adapter_medical`.
2. **Train Legal:** Load `base` + `adapter_medical` (if supported) OR mix Medical data into Legal training.
   *Better approach:* Train `adapter_legal` on base. Use logic to swap adapters based on user query intent.

### Scenario 4: Resuming Interrupted Training
1. Identify last checkpoint: `output/run_name/checkpoint-1200`.
2. Run same command with added flag: `--resume_from_checkpoint output/run_name/checkpoint-1200`.

### Scenario 5: Overriding vs. Extending
- **To Override (New Domain):** Train on NEW data only. (Old knowledge lost).
- **To Extend (Keep Old):** Train on MIXED (Old + New) data.

---

## 11. Troubleshooting & Best Practices

| Issue | Solution |
| :--- | :--- |
| **OOM (Out of Memory)** | Reduce `--per_device_train_batch_size`, increase `--gradient_accumulation_steps`, or use `--quantization_bit 4`. |
| **Loss Not Decreasing** | Check data formatting, increase learning rate, or verify tokenizer alignment. |
| **Model Forgets Previous Task** | Implement Data Mixing (Strategy A) or use separate LoRA adapters. |
| **Training Too Slow** | Enable `--flash_attn`, use DeepSpeed ZeRO-2/3, or switch to LoRA. |
| **Garbage Output** | Verify `template` matches your data format; check for overfitting (loss ~0). |

### Golden Rules
1. **Always Checkpoint:** Long runs *will* fail. Save often.
2. **Validate Early:** Generate text after 100 steps to ensure sanity.
3. **Mix Data:** Never train on a single narrow dataset if you want a general assistant.
4. **Version Everything:** Name outputs clearly (`v1`, `v2`, `exp_lr_high`).

---

**End of Guide.**  
*Ready to build the future with Nexuss AI.*
