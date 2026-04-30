# Nexuss AI Training Framework: End-to-End Deep Dive

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Training Pipeline](#core-training-pipeline)
3. [Training Stages Explained](#training-stages-explained)
4. [Checkpointing Mechanism](#checkpointing-mechanism)
5. [Versioning Strategy](#versioning-strategy)
6. [Catastrophic Forgetting Analysis](#catastrophic-forgetting-analysis)
7. [Fine-tuning Methods](#fine-tuning-methods)
8. [Data Processing Pipeline](#data-processing-pipeline)
9. [Model Loading & Adapters](#model-loading--adapters)
10. [Practical Usage Guide](#practical-usage-guide)

---

## Architecture Overview

### Directory Structure
```
/workspace/src/llamafactory/
├── train/                  # Training workflows
│   ├── pt/                # Pre-training (from scratch)
│   ├── sft/               # Supervised Fine-Tuning
│   ├── dpo/               # Direct Preference Optimization
│   ├── ppo/               # Proximal Policy Optimization
│   ├── rm/                # Reward Modeling
│   └── kto/               # KTO Alignment
├── model/                  # Model loading & adapters
├── data/                   # Data processing
├── hparams/               # Configuration arguments
├── extras/                # Utilities
└── api/                   # API endpoints
```

### Entry Points
- **`src/train.py`**: Main training entry point → calls `run_exp()` in `tuner.py`
- **`src/llamafactory/train/tuner.py`**: Orchestrates all training stages

---

## Core Training Pipeline

### Flow Diagram
```
User Command (train.py)
    ↓
run_exp() [tuner.py]
    ↓
read_args() → Parse CLI arguments
    ↓
get_train_args() → Validate & load configurations
    ↓
_training_function() → Route to specific stage
    ↓
[run_pt | run_sft | run_dpo | run_ppo | run_rm | run_kto]
    ↓
Load Tokenizer → Load Model → Process Dataset → Initialize Trainer
    ↓
trainer.train(resume_from_checkpoint=...)
    ↓
Save Checkpoints → Generate Reports
```

### Key Components

#### 1. **Argument Parsing** (`hparams/`)
- `ModelArguments`: Model path, quantization, adapter config
- `DataArguments`: Dataset paths, preprocessing options
- `TrainingArguments`: HuggingFace TrainingArguments extensions
- `FinetuningArguments`: LoRA, freeze, full-tuning parameters

#### 2. **Training Router** (`tuner.py:62-115`)
```python
if finetuning_args.stage == "pt":
    run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
elif finetuning_args.stage == "sft":
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
# ... other stages
```

---

## Training Stages Explained

### 1. Pre-training (`stage=pt`)
**Purpose**: Train language model from scratch on raw text

**Workflow** (`train/pt/workflow.py`):
```python
def run_pt(...):
    tokenizer = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, ..., stage="pt")
    model = load_model(tokenizer, model_args, finetuning_args, do_train=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = CustomTrainer(model, args, finetuning_args, data_collator, ...)
    
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()  # Saves optimizer state, scheduler, etc.
```

**Key Characteristics**:
- Uses `DataCollatorForLanguageModeling` (causal LM, no masking)
- No labels required (self-supervised)
- Supports resume from checkpoint

### 2. Supervised Fine-Tuning (`stage=sft`)
**Purpose**: Teach model to follow instructions using conversational pairs

**Workflow** (`train/sft/workflow.py`):
```python
def run_sft(...):
    dataset_module = get_dataset(template, ..., stage="sft")
    model = load_model(...)
    
    # Optional reference model for advanced losses
    if finetuning_args.use_asft_loss:
        ref_model = create_ref_model(model_args, finetuning_args)
    
    data_collator = SFTDataCollatorWith4DAttentionMask(...)
    trainer = CustomSeq2SeqTrainer(model, args, finetuning_args, ..., ref_model=ref_model)
    
    train_result = trainer.train(resume_from_checkpoint=...)
```

**Data Format Required**:
```json
{"messages": [
  {"role": "user", "content": "Question?"},
  {"role": "assistant", "content": "Answer"}
]}
```

### 3. DPO/PPO/KTO (Alignment Stages)
- **DPO**: Direct preference optimization (chosen vs rejected pairs)
- **PPO**: Reinforcement learning with reward model
- **KTO**: Kahneman-Tversky optimization (implicit preferences)

---

## Checkpointing Mechanism

### How It Works

#### 1. **Automatic Checkpoint Saving**
Controlled by `TrainingArguments`:
```python
save_steps: int = 500        # Save every N steps
save_total_limit: int = None # Keep all checkpoints (or limit)
save_safetensors: bool = True # Use safe serialization
```

**Implementation** (`transformers.Trainer` + callbacks):
- Every `save_steps`, trainer calls `on_save()` callback
- Checkpoint saved to: `{output_dir}/checkpoint-{global_step}/`

#### 2. **Checkpoint Contents**
Each checkpoint directory contains:
```
checkpoint-500/
├── model.safetensors      # Model weights
├── training_state.json    # Optimizer, scheduler, global step
├── trainer_state.json     # Training history, logs
├── optimizer.pt           # Optimizer state
├── scheduler.pt           # Learning rate scheduler
└── rng_state.pth          # Random number generator state
```

#### 3. **Resume from Checkpoint**
**Code Path** (`workflow.py:63, 140`):
```python
train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
```

**Usage**:
```bash
# Resume from specific checkpoint
python src/train.py --resume_from_checkpoint output/nexuss_ai/checkpoint-1000

# Auto-resume from latest checkpoint
python src/train.py --resume_from_checkpoint output/nexuss_ai/checkpoint-1000
```

**What Gets Restored**:
- ✅ Model weights
- ✅ Optimizer state (momentum, etc.)
- ✅ Learning rate scheduler
- ✅ Global step count
- ✅ Random state (for reproducibility)

#### 4. **Callback System** (`train/callbacks.py`)
Key callbacks for checkpointing:
- `LogCallback`: Logs training progress
- `FixValueHeadModelCallback`: Fixes reward model checkpoints
- `SaveProcessorCallback`: Saves tokenizers/processors
- `PissaConvertCallback`: Converts PiSSA adapters

---

## Versioning Strategy

### Current Architecture Support

The framework **does NOT have built-in semantic versioning** like GPT releases. Instead, it uses **directory-based versioning**:

### Manual Versioning Workflow

#### 1. **Output Directory as Version**
```bash
# Version 0.1 - Base pre-training
--output_dir output/nexuss_v0.1_pretrain

# Version 0.2 - After SFT
--output_dir output/nexuss_v0.2_sft

# Version 1.0 - After DPO alignment
--output_dir output/nexuss_v1.0_dpo
```

#### 2. **Export for Release** (`tuner.py:140-237`)
```bash
python src/export_model.py \
    --model_name_or_path output/nexuss_v1.0_dpo \
    --export_dir releases/nexuss_ai_v1.0 \
    --export_size 2 \
    --export_hub_model_id your_org/nexuss-ai-1.0
```

**Export Process**:
1. Loads model + adapters
2. Merges LoRA weights (if applicable)
3. Converts dtype (float16/bfloat16)
4. Saves in sharded format
5. Optionally pushes to HuggingFace Hub

#### 3. **Creating Standalone Releases**
For LoRA fine-tuned models:
```python
# In export_model():
if model_args.adapter_name_or_path is not None:
    # Merge adapter into base model
    model = merge_adapter(model, adapter_path)
    
model.save_pretrained(export_dir)
tokenizer.save_pretrained(export_dir)
```

**Result**: A standalone model directory ready for deployment.

### Recommended Versioning Convention
```
releases/
├── nexuss-base-v0.1/      # Raw pre-trained model
├── nexuss-instruct-v0.2/  # After SFT
├── nexuss-aligned-v1.0/   # After DPO/PPO
└── nexuss-v1.1-ft-taskA/  # Task-specific fine-tune
```

---

## Catastrophic Forgetting Analysis

### ⚠️ Critical Finding: **NOT Automatically Handled**

The architecture **does NOT** have built-in mechanisms to prevent catastrophic forgetting. This is confirmed by analyzing the training workflows:

### Evidence from Code

#### 1. **Full Fine-tuning Overwrites All Weights**
```python
# adapter.py:_setup_full_tuning()
for name, param in model.named_parameters():
    if not any(forbidden_module in name for forbidden_module in forbidden_modules):
        param.requires_grad_(True)  # ALL layers trainable
```

**Result**: Previous task knowledge is overwritten during new task training.

#### 2. **LoRA Adapters Replace Previous Adapters**
```python
# When loading adapter:
model = load_model(tokenizer, model_args, finetuning_args)
# If adapter_name_or_path is specified, it loads NEW adapter only
```

**Result**: Loading a new adapter does not preserve old adapter capabilities.

#### 3. **No Replay Buffer Implementation**
Searching the codebase:
- ❌ No `replay_buffer` parameter
- ❌ No `old_data_mixing` option
- ❌ No `elastic_weight_consolidation` (EWC)
- ❌ No `gradient_episodic_memory` (GEM)

### Consequences

**Scenario**: Train on Task A → Fine-tune on Task B
```
Before Task B training:
- Model excels at Task A (95% accuracy)

After Task B training:
- Model excels at Task B (93% accuracy)
- Model FAILS at Task A (40% accuracy) ← CATASTROPHIC FORGETTING
```

### Mitigation Strategies (Manual Implementation Required)

#### Strategy 1: **Data Mixing** (Recommended)
Combine datasets before training:
```python
# In dataset_info.json:
{
  "nexuss_mixed": {
    "file_name": "task_a_data.jsonl,task_b_data.jsonl",
    "mix_strategy": "concat"  # or "interleave"
  }
}
```

**Command**:
```bash
python src/train.py \
    --dataset task_a_data,task_b_data \
    --mix_strategy concat \
    --stage sft
```

#### Strategy 2: **Separate LoRA Adapters**
Maintain multiple adapters for different tasks:
```
adapters/
├── nexuss_task_a/   # Adapter for Task A
├── nexuss_task_b/   # Adapter for Task B
└── nexuss_base/     # Base pre-trained model
```

**Usage**:
```bash
# Switch between tasks by loading different adapters
python src/infer.py --adapter_name_or_path adapters/nexuss_task_a
python src/infer.py --adapter_name_or_path adapters/nexuss_task_b
```

#### Strategy 3: **Replay Buffer** (Manual)
Include 10-30% of old data in new training:
```python
# Create mixed dataset manually:
import random

old_data = load_jsonl("task_a_data.jsonl")
new_data = load_jsonl("task_b_data.jsonl")

# Mix: 70% new, 30% old
mixed = new_data + random.sample(old_data, k=len(new_data) // 3)
save_jsonl(mixed, "task_b_with_replay.jsonl")
```

#### Strategy 4: **Progressive Fine-tuning**
Use increasingly smaller learning rates:
```bash
# Task A: lr=1e-4
python src/train.py --learning_rate 1e-4 --dataset task_a

# Task B: lr=1e-5 (10x smaller to preserve Task A knowledge)
python src/train.py --learning_rate 1e-5 --dataset task_b
```

### Best Practices Summary

| Method | Effectiveness | Complexity | Memory Overhead |
|--------|--------------|------------|-----------------|
| Data Mixing | ⭐⭐⭐⭐⭐ | Low | None |
| Separate Adapters | ⭐⭐⭐⭐ | Medium | Low (per adapter) |
| Replay Buffer | ⭐⭐⭐⭐ | Low | Storage for old data |
| Lower LR | ⭐⭐⭐ | Low | None |
| Full Re-training | ⭐⭐⭐⭐⭐ | High | High |

---

## Fine-tuning Methods

### Supported Methods (`finetuning_type`)

#### 1. **Full Fine-tuning** (`full`)
```python
# adapter.py:_setup_full_tuning()
for param in model.parameters():
    param.requires_grad_(True)  # All parameters trainable
```

**Pros**: Maximum performance potential  
**Cons**: High memory, catastrophic forgetting risk  
**Use Case**: When you have abundant GPU memory and data

#### 2. **LoRA** (`lora`) - **DEFAULT**
```python
# LoraArguments (finetuning_args.py)
lora_rank: int = 8              # Intrinsic dimension
lora_alpha: int = 16            # Scaling factor (default: rank*2)
lora_dropout: float = 0.0       # Dropout rate
lora_target: str = "all"        # Target modules
use_dora: bool = False          # Weight-decomposed LoRA
```

**How It Works**:
```
Original: W ∈ R^(d×k)
LoRA: W + ΔW = W + BA, where B ∈ R^(d×r), A ∈ R^(r×k)
Only A and B are trainable (r << d,k)
```

**Pros**: 
- 10-100x fewer trainable parameters
- Multiple adapters can coexist
- Easy to switch tasks

**Cons**: 
- Slight performance drop vs full tuning
- Still susceptible to forgetting when merging

#### 3. **Freeze Tuning** (`freeze`)
```python
# FreezeArguments
freeze_trainable_layers: int = 2  # Train last N layers
freeze_trainable_modules: str = "all"
```

**How It Works**: Only trains top N transformer layers, freezes rest.

#### 4. **OFT** (`oft`)
Orthogonal Finetuning - preserves pre-trained knowledge better than LoRA.

### Advanced LoRA Features

#### DoRA (Weight-Decomposed LoRA)
```bash
--use_dora True
```
Decomposes weights into magnitude and direction for better stability.

#### PiSSA (Principal Singular Values Adaptation)
```bash
--pissa_init True --pissa_iter 16
```
Initializes LoRA with principal singular vectors for faster convergence.

#### LoRA+ (Different LR for A and B matrices)
```bash
--loraplus_lr_ratio 16.0  # lr_B / lr_A = 16
```

---

## Data Processing Pipeline

### Data Flow
```
Raw Dataset (JSONL)
    ↓
load_dataset() [data/loader.py]
    ↓
Pre-process based on stage
    ↓
[PretrainProcessor | SFTProcessor | PairwiseProcessor | ...]
    ↓
Apply Template (add special tokens, formatting)
    ↓
Tokenize (convert to IDs)
    ↓
Collate (batch + padding)
    ↓
Model Input
```

### Processor Types (`data/processor/`)

#### 1. **PretrainProcessor** (`pretrain.py`)
- Input: `{"text": "raw text..."}`
- Output: Continuous token sequences
- No special formatting

#### 2. **SupervisedProcessor** (`supervised.py`)
- Input: `{"messages": [...]}`
- Output: Formatted conversation with roles
- Applies chat template

#### 3. **PairwiseProcessor** (`pairwise.py`)
- Input: `{"chosen": [...], "rejected": [...]}`
- Used for DPO/RM training

### Template System (`data/template.py`)
Defines how conversations are formatted:
```python
# Example template structure
template = Template(
    prefix=["<s>"],
    user="<human>\n{content}\n",
    assistant="<bot>\n{content}</s>",
    system="<system>\n{content}\n"
)
```

**Built-in Templates**: `default`, `llama3`, `qwen`, `chatglm`, etc.

### Tiktoken Integration
Already verified working:
```python
# data/tiktoken_utils.py
from tiktoken import get_encoding

encoding = get_encoding("cl100k_base")
tokens = encoding.encode("Hello, Nexuss AI!")
```

---

## Model Loading & Adapters

### Loading Flow (`model/loader.py`)

```python
def load_model(tokenizer, model_args, finetuning_args, is_trainable):
    # 1. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        quantization_config=quant_config,  # If quantized
        ...
    )
    
    # 2. Apply adapters if specified
    if finetuning_args.finetuning_type == "lora":
        model = _setup_lora(model, finetuning_args, is_trainable)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable)
    elif finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, finetuning_args, is_trainable)
    
    # 3. Load pre-existing adapter if provided
    if model_args.adapter_name_or_path:
        model.load_adapter(adapter_path)
    
    return model
```

### Adapter Merging (`export_model()`)
When exporting, LoRA adapters are merged:
```python
# PeftModel.merge_and_unload()
merged_model = model.merge_and_unload()
merged_model.save_pretrained(export_dir)
```

This creates a standalone model without adapter dependencies.

---

## Practical Usage Guide

### Scenario 1: Train Nexuss AI from Scratch

#### Step 1: Prepare Dataset
```bash
# Create data file
cat > data/nexuss_pretrain.jsonl << EOF
{"text": "Nexuss AI is a powerful language model..."}
{"text": "It can understand and generate human-like text..."}
EOF
```

#### Step 2: Register Dataset
Edit `data/dataset_info.json`:
```json
{
  "nexuss_pretrain": {
    "file_name": "nexuss_pretrain.jsonl",
    "columns": {"prompt": "text"}
  }
}
```

#### Step 3: Run Pre-training
```bash
cd /workspace && python src/train.py \
    --stage pt \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --use_tiktoken True \
    --tiktoken_encoding cl100k_base \
    --do_train \
    --dataset nexuss_pretrain \
    --dataset_dir data \
    --template default \
    --finetuning_type full \
    --output_dir output/nexuss_base_v0.1 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --fp16
```

### Scenario 2: Fine-tune with Catastrophic Forgetting Prevention

#### Option A: Data Mixing
```bash
python src/train.py \
    --stage sft \
    --model_name_or_path output/nexuss_base_v0.1 \
    --dataset task_a_data,task_b_data \
    --mix_strategy concat \
    --finetuning_type lora \
    --output_dir output/nexuss_mixed_v0.2 \
    --learning_rate 1e-4 \
    --lora_rank 16
```

#### Option B: Separate Adapters
```bash
# Train Task A adapter
python src/train.py \
    --stage sft \
    --model_name_or_path output/nexuss_base_v0.1 \
    --dataset task_a_data \
    --finetuning_type lora \
    --output_dir output/nexuss_adapter_task_a

# Train Task B adapter (separate)
python src/train.py \
    --stage sft \
    --model_name_or_path output/nexuss_base_v0.1 \
    --dataset task_b_data \
    --finetuning_type lora \
    --output_dir output/nexuss_adapter_task_b

# Use Task A
python src/infer.py --adapter_name_or_path output/nexuss_adapter_task_a

# Use Task B
python src/infer.py --adapter_name_or_path output/nexuss_adapter_task_b
```

### Scenario 3: Resume Interrupted Training
```bash
# Find latest checkpoint
ls -lt output/nexuss_base_v0.1/checkpoint-*

# Resume
python src/train.py \
    --stage pt \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --dataset nexuss_pretrain \
    --output_dir output/nexuss_base_v0.1 \
    --resume_from_checkpoint output/nexuss_base_v0.1/checkpoint-1500
```

### Scenario 4: Export Final Model for Release
```bash
python src/export_model.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --adapter_name_or_path output/nexuss_adapter_task_a \
    --export_dir releases/nexuss-instruct-v1.0 \
    --export_size 2 \
    --export_hub_model_id tadiyos/nexuss-ai-1.0
```

---

## Key Takeaways

### ✅ Strengths of This Architecture
1. **Flexible Training Stages**: PT → SFT → DPO/PPO pipeline
2. **Multiple Fine-tuning Methods**: LoRA, full, freeze, OFT
3. **Robust Checkpointing**: Full state recovery supported
4. **Tiktoken Integration**: Verified working
5. **Adapter Support**: Easy task switching with LoRA

### ⚠️ Limitations to Address
1. **No Built-in Catastrophic Forgetting Prevention**: Must implement manually
2. **No Semantic Versioning**: Directory-based versioning only
3. **Sequential Fine-tuning Risk**: Will overwrite previous knowledge

### 🎯 Recommendations for Nexuss AI

1. **For Pre-training**:
   - Use `--stage pt` with `--finetuning_type full`
   - Set `--save_steps 500` for frequent checkpoints
   - Enable `--save_total_limit 3` to manage disk space

2. **For Fine-tuning**:
   - Use `--finetuning_type lora` for efficiency
   - **Mix datasets** instead of sequential training
   - Maintain **separate adapters** for different tasks

3. **For Production Releases**:
   - Export merged models with `export_model.py`
   - Use semantic versioning in directory names
   - Push to HuggingFace Hub for distribution

4. **To Prevent Forgetting**:
   - Always include 10-30% replay data
   - Use lower learning rates for subsequent tasks
   - Consider separate adapters per use case

---

## Appendix: Complete Command Reference

### Pre-training
```bash
python src/train.py --stage pt --model_name_or_path [BASE_MODEL] --dataset [DATASET] --output_dir [OUTPUT]
```

### Supervised Fine-tuning
```bash
python src/train.py --stage sft --model_name_or_path [PRETRAINED_MODEL] --dataset [DATASET] --finetuning_type lora
```

### DPO Alignment
```bash
python src/train.py --stage dpo --model_name_or_path [SFT_MODEL] --dataset [PREFERENCE_DATA] --ref_model [REF_MODEL]
```

### Resume Training
```bash
python src/train.py --resume_from_checkpoint [CHECKPOINT_PATH]
```

### Export Model
```bash
python src/export_model.py --model_name_or_path [MODEL] --adapter_name_or_path [ADAPTER] --export_dir [EXPORT_PATH]
```

### Inference
```bash
python src/infer.py --model_name_or_path [MODEL] --adapter_name_or_path [ADAPTER]
```

---

**Document Version**: 1.0  
**Last Updated**: Based on codebase analysis  
**Author**: Nexuss AI Development Team
