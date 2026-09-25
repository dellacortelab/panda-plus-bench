# PANDA-PLUS-Bench: Evaluating WSI-Specific Feature Collapse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dellacortelab/PANDA-PLUS-Bench/blob/main/PANDA_PLUS_Bench_Evaluation.ipynb)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PANDA--PLUS--Bench-yellow)](https://huggingface.co/datasets/dellacorte/PANDA-PLUS-Bench)

A standardized benchmark for evaluating pathology foundation models on whole-slide image (WSI)-specific feature collapse.

## 📖 Overview

PANDA-PLUS-Bench measures how well pathology foundation models distinguish between **biological features** (cancer grade) versus **slide-specific artifacts** (staining, scanner effects, etc.). Models that overfit to slide-specific features will show artificially high within-slide accuracy but poor cross-slide generalization.

**Key Metrics:**
- **Within-slide accuracy**: k-NN classification trained and tested on patches from the same slide
- **Cross-slide accuracy**: Leave-one-slide-out k-NN classification
- **Accuracy gap**: Within - Cross (higher = more feature collapse)
- **Robustness testing**: 8 augmentation conditions (baseline, color jitter, grayscale, etc.)

## 🚀 Quick Start

### Run on Google Colab (Recommended)

Click the badge above or use this link:
```
https://colab.research.google.com/github/dellacortelab/PANDA-PLUS-Bench/blob/main/PANDA_PLUS_Bench_Evaluation.ipynb
```

**What you need:**
- A Google account (free GPU available)
- Optional: HuggingFace token for gated models (UNI, Virchow)

**Runtime:** ~5-15 minutes of embedding extraction per augmentation condition on a Colab T4 GPU, plus a one-time dataset download (~2.6 GB for all 8 conditions)

### Evaluate Your Model

```python
# In the Configuration cell (Step 1), pick a preset model or choose "custom":
MODEL_CHOICE = "custom"
CUSTOM_MODEL_ID = "your-org/your-model"  # Any HuggingFace model
# or
CUSTOM_MODEL_ID = "resnet50"  # Any timm model (used if HuggingFace loading fails)

# Select augmentation conditions to test
EVAL_BASELINE = True
EVAL_COMBINED_AGGRESSIVE = True
# ... configure other augmentations

# Run all cells!
```

## 📊 Benchmark Dataset

The benchmark uses expert-annotated prostate biopsy patches derived from the PANDA dataset, with controlled augmentations:

- **3,872 patches** per augmentation condition (224×224 px)
- **9 whole slide images** (one per patient)
- **3 classes** (labels 0–2), not balanced across classes or slides
- **8 augmentation conditions**:
  - `baseline`: Original patches
  - `color_jitter`: Color variations
  - `grayscale`: Removes color information
  - `gaussian_noise`: Adds noise
  - `heavy_geometric`: Rotation, flip, scale
  - `combined_aggressive`: Multiple augmentations
  - `macenko_normalization`: Stain normalization
  - `hed_stain_augmentation`: H&E stain variation

Dataset available at: [huggingface.co/datasets/dellacorte/PANDA-PLUS-Bench](https://huggingface.co/datasets/dellacorte/PANDA-PLUS-Bench)

## 📈 Supported Models

**Preset options in the notebook:**
- Phikon / Phikon-v2 (Owkin)
- UNI (Mahmood Lab) - requires HF token
- Virchow / Virchow2 (Paige.AI) - requires HF token

**Custom models** (set `MODEL_CHOICE = "custom"`):
- HuggingFace models loadable with `transformers.AutoModel` (embeddings are taken from the CLS token)
- `timm` models by name (ResNet, EfficientNet, ViT, etc.), used as a fallback if HuggingFace loading fails

Models that need a custom loader (e.g. those distributed only via `timm`'s `hf_hub:` prefix) may require editing `load_foundation_model` in Step 3.

**Published comparison results** (`paper_results.json`) are included for 7 models: Phikon, Phikon-v2, UNI, UNI2, Virchow, Virchow2, and HistoEncoder. They currently cover the `baseline` condition only.

## 🔬 What Gets Measured

The notebook computes:

1. **Classification Accuracy**
   - Within-slide: k-NN (k=5 by default) with an 80/20 train/test split within each slide
   - Cross-slide: Leave-one-slide-out k-NN
   - Gap metric: Measures overfitting to slide identity

2. **Embedding Quality**
   - Silhouette scores for class vs. slide clustering
   - k-NN same-slide neighbor fraction (k=50)
   - Slide-ID prediction accuracy (logistic regression, 5-fold CV)
   - t-SNE visualizations colored by label and by slide

3. **Across Augmentations and Models**
   - All metrics reported per selected augmentation condition
   - Comparison against published results for the same conditions
   - Summary ratings and recommendations

## 📝 Citation

If you use PANDA-PLUS-Bench in your research, please cite:

```bibtex
@article{ebbert2025pandaplusbench,
  title={PANDA-PLUS-Bench: A Benchmark for Evaluating WSI-Specific Feature Collapse in Pathology Foundation Models},
  author={Ebbert, Joshua and Della Corte, Dennis},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please:
- Report issues or bugs via GitHub Issues
- Suggest additional augmentation conditions
- Share results from new models

## 📄 License

The code in this repository is licensed under the MIT License - see the LICENSE file for details. The dataset on HuggingFace is licensed under CC-BY-4.0.

## 🙏 Acknowledgments

- Based on the [PANDA dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
- Built with HuggingFace Datasets and PyTorch
- Inspired by robustness evaluation practices in computer vision

## 📧 Contact

- **Maintainers**: Joshua Ebbert, Dennis Della Corte
- **Lab**: [Della Corte Lab](https://github.com/dellacortelab)
- **Issues**: [GitHub Issues](https://github.com/dellacortelab/PANDA-PLUS-Bench/issues)

---

**Note:** This benchmark evaluates foundation models on a specific task. Results may not generalize to all downstream applications. Use in combination with other evaluation metrics for comprehensive model assessment.
