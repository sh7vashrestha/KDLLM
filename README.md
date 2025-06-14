# KDLLM: Knowledge Distillation for Compressed and Copyright-Safe Large Language Model Sharing

KDLLM is a novel framework that leverages **knowledge distillation** to enable the creation of compact, high-performing large language models (LLMs) while safeguarding the intellectual property (IP) of the original model. This framework provides a solution for communication-efficient and copyright-safe model sharing.

## 🚀 Features

- **Performance Preservation**: Student models achieve over 93% of the original model's accuracy.
- **High Compression**: Up to **87% reduction** in model size.
- **IP Protection**: Obfuscates teacher model architecture and parameters.
- **Flexible Distillation Losses**: Supports both **KL Divergence** and **Wasserstein Distance**.
- **Efficient Deployment**: Ideal for low-resource or privacy-sensitive environments.

## 📄 Overview

- **Paper Title**: *KDLLM: Knowledge Distillation for Compressed and Copyright-Safe Large Language Model Sharing*
- **Published In**: Tsinghua Science and Technology, 2025
- **Authors**: Shiva Shrestha et al.
- **Keywords**: Knowledge Distillation, LLMs, Copyright Protection, Model Compression

## 📘 Architecture

- **Teacher Model**: BERT-Base (110M parameters)
- **Student Model**: Custom BERT variant (13.9M parameters)
- **Distillation Losses**:
  - KL Divergence (KLD)
  - Wasserstein Distance (WD)

## 📊 Performance Summary

| Model                 | Accuracy | F1 Score | Size    | Compression |
|----------------------|----------|----------|---------|-------------|
| BERT-Base (Teacher)  | 92.40%   | 92.44%   | 418 MB  | -           |
| KDLLM + KLD          | 86.30%   | 86.23%   | 54 MB   | 7.74x       |
| KDLLM + WD           | 85.91%   | 86.11%   | 54 MB   | 7.74x       |
| Watermarked Teacher  | 81.61%   | 84.37%   | 418 MB  | -           |
| Lexical Baseline     | 69.83%   | 70.00%   | 55 MB   | -           |

## 🧪 Experimental Setup

- **Dataset**: IMDb movie review sentiment classification
- **Environment**: Google Colab (Tesla T4 GPU)
- **Training**: 10 epochs, batch size 16, AdamW optimizer, LR = 2e-5
- **Best temperature for distillation**: T = 10

## 🔐 Copyright and IP Protection

KDLLM introduces:

- **Obfuscated Architecture**: Student model can differ structurally from the teacher.
- **Behavior-Only Transfer**: Distills only the output behavior (soft labels), not internal weights.
- **Legal Safety**: Reduces risk of IP violation through reverse engineering.

## 📦 Applications

- Safe model publication in research and academia
- Commercial model deployment with copyright assurance
- Efficient LLM use in bandwidth-limited or edge devices

## 🔮 Future Work

- Extend KDLLM to generative tasks (e.g., translation, summarization)
- Explore cryptographic watermarking integration
- Apply to federated and multimodal learning contexts

## 📚 Citation

```bibtex
@article{shrestha2025kdllm,
  title={KDLLM: Knowledge Distillation for Compressed and Copyright-Safe Large Language Model Sharing},
  author={Shiva Shrestha et al.},
  journal={Tsinghua Science and Technology},
  year={2025}
}
```

## 📬 Contact

For more details, refer to the publication at [Tsinghua Science and Technology Journal](https://mc03.manuscriptcentral.com/tst).

---

> ⚠️ This repository is intended for educational and research purposes. For commercial licensing or usage, ensure compliance with applicable intellectual property regulations.
