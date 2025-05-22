## DynSE-BDResNet：Dual-Path Dynamically Regularized Residual Learning with Bayesian-Channel Co-Attention in Heart Murmur Detection (Machine Learning in Audio Signal Processing Spring 2025)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**1. Dual-Path Bayesian Regularization Architecture (DBRes-Net)**
 We propose a dual stochastic regularization framework to mitigate feature co-adaptation while preserving diagnostically critical patterns. The architecture integrates (1) *position-adaptive Bayesian dropout layers* that enforce persistent stochastic masking across both training and inference phases, maintaining epistemic uncertainty quantification through depth-dependent probability modulation. This contrasts with conventional dropout by preventing deterministic feature suppression during inference. (2) A *hybrid channel-spatial attention gate* probabilistically reweights feature responses through synergistic channel-wise excitation (via squeeze-and-excitation) and spatial saliency mapping (via max-mean pooling fusion). The dual attention mechanism operates as a nonlinear gating function, selectively amplifying pathologically salient biomarkers while inducing stochastic feature suppression in non-informative regions. This dual randomization strategy synergistically balances model generalization with interpretable feature prioritization.

**2. Adaptive Kernel Attention (AKA) for Multiscale Feature Recalibration**
 Our attention mechanism introduces dynamic receptive field adaptation to enhance pathological pattern detection. The channel attention submodule autonomously optimizes convolutional kernel dimensions (3/5/7) through prime-number-based selection rules, enabling scale-adaptive context aggregation tailored to feature map cardinality. Concurrently, the spatial submodule implements differential pooling fusion, combining max-pooling’s lesion-localization advantages with average-pooling’s global context preservation. The compounded attention weights are computed through a cascaded nonlinear transformation (quantified via gradient-weighted class activation mapping). This multiscale recalibration mechanism enables context-aware refinement of both local acoustic anomalies and global hemodynamic patterns.

**3. Heterogeneous Multimodal Decision Fusion**
 We establish an end-to-end fusion pipeline that jointly optimizes deep auditory representations and shallow clinical biomarkers. The framework hierarchically combines (1) high-dimensional latent features from the ResNet’s penultimate layer, encoding nonlinear acoustic texture patterns, (2) 18 clinical covariates (demographic/auscultation metadata), and (3) 23 handcrafted complexity descriptors (multiscale entropy, spectral-temporal variability). A gradient-boosted ensemble learner with adaptive tree-depth modulation (max_depth=8) is employed to model cross-modal interactions, dynamically weighting feature contributions through impurity-based importance scoring. The architecture’s decision boundaries are further regularized via Bayesian hyperparameter optimization, ensuring robustness to feature scale disparities.

| Rank | Model                             | Weighted Accuracy (↑) | Weighted Accuracy (Validation) | Weighted Accuracy (Cross-Val) |
| ---- | --------------------------------- | --------------------- | ------------------------------ | ----------------------------- |
| 1    | **DynSE-BDResNet—XGBoost Fusion** | **0.914**             | **0.788**                      | **0.753**                     |
| 2    | M2D                               | 0.832                 | 0.713                          | -                             |
| 3    | DBResNet                          | 0.771                 | 0.768                          | -                             |
| 4    | Inception Time                    | 0.593                 | 0.522                          | 0.497±0.083                   |

## Data

The challenge dataset can be downloaded via this [link](https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip)

## Dependencies

- At least Python 3.9;
- [PyTorch (torch, torchvision)](https://github.com/pytorch/pytorch/) for neural network architecture and training;
- [XGBoost](https://github.com/dmlc/xgboost) for xgboost;
- [Librosa](https://github.com/librosa/librosa) for audio processing and feature extraction;

### Install using conda on cuda device

```shell
conda create -n myenv python=3.9
conda activate myenv
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running Experiments

You can run all steps using the template files in `run_files`.`

A full experiment consists of four steps:

1. Splitting the data into stratified train, test, and validation sets (`data_split.py`).
2. Training the Bayesian ResNets on their respective binary classification tasks (`train_resnet.py`).
3. Calculating and evaluating the output from DBRes (`dbres.py`).
4. Calculating and evaluating the output from DBRes with XGBoost integration (`xgboost_integration.py`).

These steps can be run independently using the relevant script, or sequentially using `main.py`,

```shell
python main.py
```

## Experimental Environment

- **GPU**: NVIDIA GeForce RTX 4090 (24GB GDDR6X) × 1  
- **CPU**: 16 virtualized cores (Intel® Xeon® Platinum 8352V @ 2.10GHz, Cascade Lake)  
- **RAM**: 120GB DDR4 ECC Memory  
- **Storage**: 30GB NVMe SSD System Partition  
The RTX 4090's third-generation RT cores and 24GB VRAM particularly accelerate attention-based feature recalibration modules through tensor core optimization.

## Inspiration

This work builds upon the dual-branch Bayesian framework introduced by [Walker et al. (CinC 2022)](https://cinc.org/archives/2022/pdf/CinC2022-355.pdf) and their [original implementation](https://github.com/Benjamin-Walker/heart-murmur-detection), extending the methodology through adaptive spectral attention mechanisms and hybrid clinical-acoustic feature fusion.  
