# Enterprise-Grade LLaMA-Factory Training Guide

## 1. Introduction

This guide provides comprehensive instructions for leveraging the cleaned LLaMA-Factory codebase for enterprise-grade large language model (LLM) training. It covers training from scratch, fine-tuning with custom datasets, and reinforcement learning techniques, along with strategies for versioning and mitigating catastrophic forgetting.

LLaMA-Factory is a powerful and flexible framework built upon the Hugging Face Transformers library, designed to streamline the training and deployment of LLMs. This enterprise-focused version aims to provide a robust and maintainable foundation for developing custom LLMs tailored to specific business needs.

## 2. Training from Scratch (Blank Slate)

Training an LLM from scratch involves initializing a model with random weights and training it on a vast corpus of text data. This process is computationally intensive and requires significant resources.

### 2.1 Prerequisites

*   **Hardware**: High-performance GPUs (e.g., NVIDIA A100, H100) with substantial VRAM. Distributed training across multiple nodes is often necessary.
*   **Software**: Python 3.10+, PyTorch (latest stable version), CUDA Toolkit, and the dependencies listed in `requirements.txt`.
*   **Data**: A large, diverse, and high-quality text corpus. This could include web crawls, books, articles, and domain-specific texts. Data cleaning and preprocessing are crucial.

### 2.2 Data Preparation

Your blank slate dataset should be in a format compatible with LLaMA-Factory's data loaders. A common format is JSON Lines, where each line represents a training example.

```json
{"text": "Your first training sentence."}
{"text": "Another example sentence for training."}
```

Store your dataset in the `data/` directory or a specified path. You will need to create a `dataset_info.json` file to register your dataset.

```json
{
  "your_blank_slate_dataset": {
    "file_name": "your_blank_slate_dataset.jsonl",
    "columns": {
      "prompt": "text"
    }
  }
}
```

### 2.3 Configuration for Training

Training from scratch requires careful configuration of model architecture, tokenizer, optimizer, learning rate schedule, and training parameters. You will typically use a configuration file (e.g., YAML) to define these settings.

Key parameters include:

*   `model_name_or_path`: Set to `None` or a placeholder for training from scratch.
*   `tokenizer_name_or_path`: Path to a pre-trained tokenizer or configuration for training a new one.
*   `dataset`: Name of your blank slate dataset.
*   `num_train_epochs`: Number of training epochs.
*   `per_device_train_batch_size`: Batch size per GPU.
*   `gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.
*   `learning_rate`: Initial learning rate.
*   `fp16` or `bf16`: Enable mixed precision training for efficiency.
*   `output_dir`: Directory to save model checkpoints.

### 2.4 Example Training Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/llamafactory/train/sft.py \
    --model_name_or_path path/to/your/tokenizer \
    --do_train \
    --dataset your_blank_slate_dataset \
    --dataset_dir data/ \
    --template default \
    --finetuning_type full \
    --output_dir path/to/your/output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16
```

**Note**: For training truly from scratch, you would typically define a new model architecture (e.g., using `transformers.AutoModelForCausalLM.from_config`) and a new tokenizer. The above command assumes you might start with a tokenizer and build a model from scratch based on its vocabulary size, or fine-tune a very small base model as a starting point for a larger architecture.

## 3. Fine-tuning from Your Own Data

Fine-tuning adapts a pre-trained LLM to a specific task or domain using a smaller, task-specific dataset. This is a more common and resource-efficient approach than training from scratch.

### 3.1 Data Preparation for Fine-tuning

Your custom dataset for fine-tuning should be in a conversational or instruction-following format. LLaMA-Factory supports various formats, including JSON Lines with `prompt` and `response` fields, or `messages` for chat formats.

**Instruction-following format (JSON Lines):**

```json
{"instruction": "Summarize the following text:", "input": "[TEXT]", "output": "[SUMMARY]"}
{"instruction": "Translate this to French:", "input": "Hello", "output": "Bonjour"}
```

**Chat format (JSON Lines with `messages`):**

```json
{"messages": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I'm doing great! How can I help you today?"}]}
```

Create a `dataset_info.json` entry for your custom dataset:

```json
{
  "your_custom_finetune_dataset": {
    "file_name": "your_custom_finetune_dataset.jsonl",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### 3.2 Configuration for Fine-tuning

Fine-tuning configurations are similar to training from scratch, but you will specify a pre-trained model.

Key parameters:

*   `model_name_or_path`: Path to your pre-trained base model (e.g., `meta-llama/Llama-2-7b-hf`).
*   `dataset`: Name of your custom fine-tuning dataset.
*   `finetuning_type`: Choose `lora` (Low-Rank Adaptation) for efficient fine-tuning, `full` for full model fine-tuning, or `freeze`.
*   Other parameters like `num_train_epochs`, `per_device_train_batch_size`, `learning_rate`, `output_dir`.

### 3.3 Example Fine-tuning Command (LoRA)

```bash
CUDA_VISIBLE_DEVICES=0 python src/llamafactory/train/sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --do_train \
    --dataset your_custom_finetune_dataset \
    --dataset_dir data/ \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --output_dir path/to/your/finetuned_model \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16
```

## 4. Reinforcement Learning (RLHF/DPO/KTO)

Reinforcement Learning from Human Feedback (RLHF) and its variants like Direct Preference Optimization (DPO) and KTO (Kahneman-Tversky Optimization) are crucial for aligning LLMs with human preferences and instructions.

### 4.1 Data Preparation for RL

RL methods require preference data, typically consisting of prompts and multiple model responses, along with human judgments indicating which response is preferred.

**DPO/KTO format (JSON Lines):**

```json
{"prompt": "What is the capital of France?", "chosen": "Paris is the capital of France.", "rejected": "The capital of France is Berlin."}
```

Create a `dataset_info.json` entry for your RL dataset:

```json
{
  "your_rl_dataset": {
    "file_name": "your_rl_dataset.jsonl",
    "columns": {
      "prompt": "prompt",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

### 4.2 Configuration for RL Training

RL training involves a policy model (the LLM being optimized) and often a reference model (a frozen version of the policy model before RL). Key parameters include:

*   `model_name_or_path`: Path to your SFT-tuned model.
*   `ref_model`: Path to the reference model (often the same as `model_name_or_path` for DPO/KTO).
*   `dataset`: Name of your RL dataset.
*   `dpo_beta` or `kto_beta`: Hyperparameters for DPO/KTO loss.
*   Other training parameters.

### 4.3 Example DPO Training Command

```bash
CUDA_VISIBLE_DEVICES=0 python src/llamafactory/train/dpo.py \
    --model_name_or_path path/to/your/sft_model \
    --ref_model path/to/your/sft_model \
    --do_train \
    --dataset your_rl_dataset \
    --dataset_dir data/ \
    --template default \
    --finetuning_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --output_dir path/to/your/dpo_model \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --plot_loss \
    --fp16 \
    --dpo_beta 0.1
```

## 5. Catastrophic Forgetting and Versioning

**Catastrophic forgetting** occurs when a neural network forgets previously learned information upon learning new information. In LLMs, this can manifest as a loss of general capabilities after fine-tuning on a specific task.

### 5.1 Mitigating Catastrophic Forgetting

*   **Continual Learning**: Employ techniques like Elastic Weight Consolidation (EWC) or Learning without Forgetting (LwF) to regularize important parameters from previous tasks.
*   **Replay Buffers**: Mix a small portion of data from previous tasks with the new training data during fine-tuning.
*   **Parameter-Efficient Fine-Tuning (PEFT)**: Methods like LoRA (used in LLaMA-Factory) only update a small subset of parameters, significantly reducing the risk of catastrophic forgetting compared to full fine-tuning.
*   **Curriculum Learning**: Gradually introduce new tasks or data, building upon previously acquired knowledge.

### 5.2 Model Versioning and Knowledge Retention

To manage model evolution and ensure knowledge retention, adopt a robust versioning strategy similar to how large AI companies manage their models.

1.  **Semantic Versioning**: Use a versioning scheme (e.g., `v1.0.0`, `v1.1.0`, `v2.0.0`) for your trained models. Major versions for significant architectural changes or retraining from scratch, minor versions for major fine-tuning updates, and patch versions for minor bug fixes or small data updates.
2.  **Immutable Model Artifacts**: Once a model version is trained and validated, package it as an immutable artifact (e.g., Hugging Face model hub format, ONNX, or a custom format). Store these artifacts in a secure, versioned storage system (e.g., S3, Google Cloud Storage).
3.  **Checkpointing**: During training, regularly save checkpoints. These checkpoints can be used for recovery or for creating intermediate model versions.
4.  **Evaluation Suites**: Maintain a comprehensive suite of evaluation benchmarks for each model version. This allows for quantitative comparison and ensures that new versions do not degrade performance on critical tasks.
5.  **A/B Testing and Canary Deployments**: Before fully deploying a new model version, conduct A/B tests or canary deployments to observe its performance in a production environment with a small subset of users.
6.  **Rollback Strategy**: Always have a clear rollback strategy to revert to a previous stable model version if a new deployment introduces regressions.

## 6. Enterprise-Grade Considerations

### 6.1 Scalability and Distributed Training

LLaMA-Factory supports distributed training using libraries like DeepSpeed and Accelerate. This is essential for training large models on massive datasets.

*   **DeepSpeed**: Provides optimizations for memory efficiency and speed, including ZeRO (Zero Redundancy Optimizer) for large model training.
*   **Hugging Face Accelerate**: Simplifies distributed training setup across various environments (single-GPU, multi-GPU, multi-node).

Example command for distributed training with Accelerate:

```bash
accelerate launch --config_file path/to/accelerate_config.yaml src/llamafactory/train/sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --do_train \
    --dataset your_custom_finetune_dataset \
    --dataset_dir data/ \
    --template default \
    --finetuning_type lora \
    --output_dir path/to/your/finetuned_model \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --plot_loss \
    --fp16
```

### 6.2 Security and Data Privacy

*   **Access Control**: Implement strict access controls for training data, model checkpoints, and deployment endpoints.
*   **Data Anonymization/Pseudonymization**: Ensure sensitive data is properly anonymized or pseudonymized before training.
*   **Secure Infrastructure**: Train models on secure, isolated infrastructure (e.g., private cloud, on-premise).
*   **Model Auditing**: Regularly audit models for biases, fairness, and potential data leakage.

### 6.3 Monitoring and Logging

Integrate with enterprise-grade monitoring and logging solutions (e.g., Weights & Biases, MLflow, Prometheus, Grafana) to track training progress, resource utilization, and model performance.

### 6.4 Reproducibility

*   **Version Control**: Use Git for versioning code, configurations, and `dataset_info.json`.
*   **Dependency Management**: Pin all dependencies using `requirements.txt` or `conda.yaml`.
*   **Seed Everything**: Set random seeds for all components (PyTorch, NumPy, transformers) to ensure reproducibility of experiments.

### 6.5 Cost Optimization

*   **PEFT Methods**: Utilize LoRA or other PEFT techniques to reduce computational costs during fine-tuning.
*   **Mixed Precision Training**: Enable `fp16` or `bf16` for faster training and reduced memory footprint.
*   **Spot Instances**: Leverage spot instances on cloud providers for non-critical training jobs to reduce costs.
*   **Efficient Data Loading**: Optimize data loading pipelines to minimize I/O bottlenecks.

## 7. Cleaned Codebase Structure Overview

The `llama_factory_clean` directory now contains the essential components for LLM training and inference, stripped of unnecessary examples, documentation, and test files to provide a focused enterprise-grade starting point.

```
llama_factory_clean/
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md
├── data/                 # Contains dataset_info.json and custom datasets
│   ├── dataset_info.json
│   └── your_custom_dataset.jsonl
├── examples/             # Cleaned examples for reference
│   ├── alpaca_en_demo.json
│   └── ...
├── pyproject.toml
├── requirements/         # Dependency files
│   └── requirements.txt
├── scripts/              # Utility scripts
│   ├── accelerate/
│   ├── deepspeed/
│   └── ...
└── src/                  # Core LLaMA-Factory source code
    └── llamafactory/
        ├── __init__.py
        ├── api/
        ├── chat/
        ├── cli.py
        ├── data/
        ├── eval/
        ├── extras/
        ├── hparams/
        ├── launcher.py
        ├── model/
        ├── third_party/
        └── train/        # Training scripts (sft.py, dpo.py, etc.)
```

This structure emphasizes clarity and maintainability, allowing developers to quickly navigate and extend the framework for their specific needs.

---

**Author**: Manus AI
**Date**: April 27, 2026
