# PANDA-PLUS-Bench: Evaluating WSI-Specific Feature Collapse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dellacortelab/PANDA-PLUS-Bench/blob/main/PANDA_PLUS_Bench_Evaluation.ipynb)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PANDA--PLUS--Bench-yellow)](https://huggingface.co/datasets/dellacortelab/PANDA-PLUS-Bench)

A standardized benchmark for evaluating pathology foundation models on whole-slide image (WSI)-specific feature collapse.

## 📖 Overview

PANDA-PLUS-Bench measures how well pathology foundation models distinguish between **biological features** (cancer grade) versus **slide-specific artifacts** (staining, scanner effects, etc.). Models that overfit to slide-specific features will show artificially high within-slide accuracy but poor cross-slide generalization.

**Key Metrics:**
- **Within-slide accuracy**: Classification using patches from the same slide
- **Cross-slide accuracy**: Leave-one-slide-out classification  
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

**Runtime:** ~15-30 minutes per augmentation condition on Colab GPU (T4)

### Evaluate Your Model

```python
# In the Colab notebook, simply specify your model:
MODEL_ID = "your-org/your-model"  # Any HuggingFace model
# or
MODEL_ID = "resnet50"  # Any timm model

# Select augmentation conditions to test
EVAL_BASELINE = True
EVAL_COMBINED_AGGRESSIVE = True
# ... configure other augmentations

# Run all cells!
```

## 📊 Benchmark Dataset

The benchmark uses the PANDA dataset with controlled augmentations:

- **5,000 patches** per augmentation condition
- **50 whole slides** from PANDA validation set
- **Balanced classes**: ISUP grades 0-5
- **8 augmentation conditions**:
  - `baseline`: Original patches
  - `color_jitter`: Color variations
  - `grayscale`: Removes color information
  - `gaussian_noise`: Adds noise
  - `heavy_geometric`: Rotation, flip, scale
  - `combined_aggressive`: Multiple augmentations
  - `macenko_normalization`: Stain normalization
  - `hed_stain_augmentation`: H&E stain variation

Dataset available at: [huggingface.co/datasets/dellacortelab/PANDA-PLUS-Bench](https://huggingface.co/datasets/dellacortelab/PANDA-PLUS-Bench)

## 📈 Supported Models

The evaluation notebook automatically supports:

**Pathology Foundation Models:**
- Phikon / Phikon-v2 (Owkin)
- UNI / UNI-v2 (Mahmood Lab) - requires HF token
- Virchow / Virchow2 (Paige.AI) - requires HF token
- H-Optimus-0 (Bioptimus)
- Gigapath (Microsoft)

**Generic Models:**
- Any HuggingFace `transformers` model
- Any `timm` model (ResNet, EfficientNet, ViT, etc.)

**Custom Models:**
- Specify any HuggingFace model ID
- Works with models using standard transformer architectures

## 🔬 What Gets Measured

The notebook computes:

1. **Classification Accuracy**
   - Within-slide: Train/test on same slide patches
   - Cross-slide: Leave-one-slide-out validation
   - Gap metric: Measures overfitting to slide identity

2. **Embedding Quality** (optional)
   - Silhouette scores for class vs. slide clustering
   - k-NN neighbor analysis
   - Confusion attribution metrics

3. **Robustness Analysis**
   - Performance degradation across augmentations
   - Comparison to baseline condition
   - Relative robustness scores

## 📝 Citation

If you use PANDA-PLUS-Bench in your research, please cite:

```bibtex
@article{ebbert2025pandaplusbench,
  title={PANDA-PLUS-Bench: A Benchmark for Evaluating WSI-Specific Feature Collapse in Pathology Foundation Models},
  author={Ebbert, Jacob and Della Corte, Dennis},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please:
- Report issues or bugs via GitHub Issues
- Suggest additional augmentation conditions
- Share results from new models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the [PANDA dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
- Built with HuggingFace Datasets and PyTorch
- Inspired by robustness evaluation practices in computer vision

## 📧 Contact

- **Maintainers**: Jacob Ebbert, Dennis Della Corte
- **Lab**: [Della Corte Lab](https://github.com/dellacortelab)
- **Issues**: [GitHub Issues](https://github.com/dellacortelab/PANDA-PLUS-Bench/issues)

---

**Note:** This benchmark evaluates foundation models on a specific task. Results may not generalize to all downstream applications. Use in combination with other evaluation metrics for comprehensive model assessment.
