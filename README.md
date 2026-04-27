# LLaMA-Factory Enterprise: Blank Slate Training Framework

This repository provides an enterprise-grade, cleaned, and streamlined version of LLaMA-Factory, specifically tailored for **training large language models (LLMs) from a blank slate (from scratch)**. It removes unnecessary examples, pre-trained model references, and jargon to offer a focused foundation for developing custom LLMs.

## Key Features

*   **Blank Slate Training**: Comprehensive support for initializing and training LLMs from random weights. Supports multiple data formats including **JSON** and **Parquet** for large-scale training.
*   **Custom Data Fine-tuning**: Tools and guidance for fine-tuning models on your proprietary datasets.
*   **Reinforcement Learning**: Integration with DPO/KTO for aligning models with human preferences.
*   **Catastrophic Forgetting Mitigation**: Strategies, primarily through Parameter-Efficient Fine-Tuning (PEFT) like LoRA, to preserve learned knowledge across training iterations.
*   **Enterprise-Grade Practices**: Includes robust model versioning, evaluation suites, and deployment considerations.

## Getting Started

Refer to the `training_guide.md` for detailed instructions on:

1.  **Setting up your environment**.
2.  **Preparing your datasets** for blank-slate training and fine-tuning.
3.  **Executing training commands** for various stages (from scratch, fine-tuning, RL).
4.  **Implementing model versioning and release strategies**.
5.  **Understanding catastrophic forgetting mitigation** techniques.

## Repository Structure

```
llama-factory-enterprise/
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md                 # This file
├── training_guide.md         # Comprehensive training and versioning guide
├── data/                     # Placeholder for dataset_info.json and custom datasets
├── pyproject.toml
├── requirements/             # Project dependencies
├── scripts/                  # Utility scripts
└── src/                      # Core LLaMA-Factory source code
```

## Contribution

This framework is designed as a starting point for enterprise LLM development. Contributions that enhance its blank-slate training capabilities, add new enterprise features, or improve documentation are welcome.

---

**Author**: Nexuss AI
**Date**: April 27, 2026
