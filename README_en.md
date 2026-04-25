# Children Speech Emotion Recognition

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

## 📋 Project Overview

This project focuses on **Children Speech Emotion Recognition** using the C-BESD dataset. The goal is to classify children's speech signals into six emotional categories (ANGER, DISGUST, FEAR, HAPPY, NEUTRAL, SAD) through acoustic feature analysis and deep learning models.

### Key Features
- **Feature Engineering**: MFCC, Chroma STFT, ZCR, RMS, Mel-spectrogram, totaling 94-dimensional features with data augmentation
- **Model Architectures**: DrseCNN (core model), CNN, BiLSTM, Transformer, SigWavNet variants
- **Training Pipeline**: Reproducible training and evaluation scripts
- **Experiment Documentation**: Complete experimental results and visualizations

## 📊 Results Summary

| Model | Best Accuracy | Macro F1 | Parameters |
|-------|---------------|----------|------------|
| DrseCNN (Full) | 0.8599 | 0.8506 | ~45.5M |
| CNN | 0.7214 | - | ~5.8M |
| SigWavNet | 0.7318 | - | ~16.9M |
| Transformer | 0.6948 | - | ~2.0M |

**Note**: The best accuracy recorded in early experiments was about 0.86, but since random seeds were not fixed at that time, this result may not be fully reproducible. This repository uses **0.8599 accuracy and 0.8506 Macro F1** as the main reproducible reported result.

## 🏗️ Project Structure

```
children-speech-emotion-recognition/
├── src/                    # Source code
│   ├── data/              # Dataset and feature extraction
│   ├── models/            # Model definitions
│   ├── training/          # Training scripts
│   └── utils/             # Utility functions
├── experiments/           # Experiment records
│   ├── configs/          # Configuration files
│   ├── results/          # Result files (CSV, logs)
│   └── visualizations/   # Training curves
├── docs/                 # Documentation
├── assets/               # Images and figures
├── checkpoints/          # Model weights (not included)
├── data/                 # Dataset (not included)
└── legacy/               # Historical code (not maintained)
```

## 💻 Core Code Functions

| File Path | Main Function | Description |
|-----------|---------------|-------------|
| `src/training/train.py` | Main training script | Contains training loop, evaluation functions, command-line argument parsing, and model saving |
| `src/data/dataset.py` | Data loading and feature extraction | Implements audio feature extraction (MFCC, Chroma, etc.), data augmentation (noise, stretch, pitch), and data loaders |
| `src/models/models.py` | Model definitions | Contains implementations of various model architectures: DrseCNN, CNN, BiLSTM, Transformer, etc. |
| `src/data/statistics.py` | Dataset statistical analysis | Provides basic statistical analysis and visualization of the dataset |
| `experiments/configs/README.md` | Experiment configuration documentation | Records experiment parameter configurations and hyperparameter tuning guidelines |
| `docs/model-structure.md` | Model architecture documentation | Detailed explanation of DrseCNN and other model architectures and implementation details |

**Main code file relationships**:
1. **Data flow**: `dataset.py` → Extract features → Loaded by `train.py` for training
2. **Model flow**: `models.py` defines models → `train.py` trains and evaluates
3. **Experiment flow**: Configure parameters → Train model → Record results → Analyze and visualize

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage
1. **Prepare data**: Obtain the C-BESD dataset and place it in `data/raw/` directory (not included in this repository)
2. **Extract features**: Run `python src/data/statistics.py` for dataset analysis
3. **Train model**: Run `python src/training/train.py` (requires data files)
4. **Evaluate**: Check `experiments/results/` for evaluation metrics

**Important**: This repository contains only code structure and documentation. To run the code, you need:
- C-BESD dataset (contact dataset owners for access and licensing)
- Extracted feature files (.npy) or raw audio files
- Sufficient GPU memory for model training

## 📚 Documentation

- [Project Summary](docs/project-summary.md) - Project background, objectives, and outcomes summary
- [Dataset Documentation](docs/dataset.md) - C-BESD dataset detailed information
- [Feature Engineering](docs/feature-engineering.md) - Acoustic features and data augmentation techniques
- [Model Architecture](docs/model-structure.md) - DrseCNN and other models detailed description
- [Experiment Results](docs/experiment-results.md) - Detailed experiment records and analysis
- [Reproducibility Notes](docs/reproducibility-notes.md) - Reproducibility challenges and improvement plans
- [Known Issues](docs/known-issues.md) - Implementation limitations and areas for improvement
- [Interview Notes](docs/interview-notes.md) - Project discussion points

## 🔧 Technical Details

### Feature Extraction
Extract 94-dimensional features from 2.5-second 16kHz audio:
- ZCR (1-dim)
- Chroma STFT (12-dim)
- MFCC (40-dim)
- RMS (1-dim)
- Mel-spectrogram (40-dim)

### Data Augmentation
Each sample generates 3 versions:
1. Original features
2. Noise-added version
3. Time-stretched + pitch-shifted version

### Core Model: DrseCNN
Deep Residual Squeeze-Excitation Convolutional Network, designed for children's speech:
- **Residual Connections**: Mitigate gradient vanishing, support deep networks
- **SE Attention**: Channel attention mechanism, enhance important features
- **Multi-stage Convolution**: Hierarchical feature extraction

## ⚠️ Limitations and Considerations

### Technical Limitations
- **Small-scale Dataset**: C-BESD has limited samples per emotion category
- **Reproducibility Challenges**: Early experiments lacked fixed random seeds
- **Computational Requirements**: DrseCNN requires significant GPU memory
- **Path Dependencies**: Code requires users to set dataset paths manually

### Usage Limitations
- **Dataset Licensing**: C-BESD dataset has usage restrictions, must comply with relevant agreements
- **Code References**: Some implementations reference open-source projects and team collaboration results
- **Academic Use Only**: For research and learning purposes only, not for clinical diagnosis or commercial deployment

## 🤝 Contribution and Collaboration

This project is an undergraduate thesis / research practice project. Some implementation details reference open-source code and were developed through team collaboration.

### Participation Contributions
- Participated in analysis of children's speech acoustic characteristics
- Organized multi-feature fusion schemes
- Participated in DrseCNN model structure iteration and understanding
- Participated in model training, comparative experiments, and ablation analysis
- Organized experimental results, thesis materials, and project documentation
- Reorganized project repository for GitHub presentation and technical communication

### Acknowledgments
- C-BESD dataset providers
- Open-source speech emotion recognition projects
- Research group members for discussions and debugging assistance

## 📄 License and Disclaimer

### License Information
The contents of this repository follow the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license:
- **Attribution**: Please credit this project when using it
- **NonCommercial**: Commercial use is prohibited

### Important Disclaimer
1. **Dataset Restrictions**: C-BESD dataset is subject to original licensing constraints, users must obtain it independently and comply with relevant agreements
2. **Code References**: Some code references open-source implementations, please comply with original project license terms
3. **Academic Integrity**: When using this project's content, please cite appropriately and respect intellectual property
4. **Usage Risks**: This code is for research purposes, does not guarantee complete absence of errors, users assume their own risks
5. **Non-Medical Use**: Models are for academic research only, not for clinical diagnosis or medical decision-making

### Citation Suggestion
If citing this project in academic work, please refer to:
```
[Project Name]. Children Speech Emotion Recognition - Lightweight Public Version. GitHub repository. 2026.
```

---
**Final Note**: This project demonstrates the full process from experimental exploration to code organization, with emphasis on research methodology and technical implementation exchange, rather than data or weight distribution.