
# Benchmarking Deepfake Detection Methods

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

### Overview

This project presents a comparative benchmarking study of representative
deepfake detection architectures inspired by the paper:

> **Towards Benchmarking and Evaluating Deepfake Detection**\
> Jingyi Deng *et al.*, IEEE Transactions on Dependable and Secure
> Computing, 2024.

Rather than reproducing the original benchmark containing thirteen
detectors, this project implements four representative architectures
spanning different deepfake detection paradigms and evaluates them under
a unified training and testing pipeline.

The benchmark compares:

| Model | Detection Paradigm |
|:------|:-------------------|
| Meso4 | Lightweight CNN |
| Xception | Deep CNN baseline |
| Patch ResNet | Patch-based CNN |
| Multiple Attention (M2TR-inspired) | Attention-based |

The project evaluates not only classification performance but also
computational efficiency, enabling practical comparisons for real-world
deployment.

------------------------------------------------------------------------

# Project Objectives

-   Implement representative deepfake detection architectures.
-   Train every model under a unified pipeline.
-   Evaluate using identical preprocessing and test conditions.
-   Compare detection performance and computational efficiency.
-   Analyze trade-offs between accuracy and deployment cost.

------------------------------------------------------------------------

## Repository Structure

```text
.
├── Report
├── Original Paper
├── data/
│   └── dataset.py
├── models/
│   ├── meso4.py
│   ├── xception.py
│   ├── patch_resnet.py
│   └── multiple_attention.py
├── train.py
├── benchmark.ipynb
├── test.py
├── weights/
├── results/
└── benchmark_output/
```

------------------------------------------------------------------------

# Models Evaluated

## Meso4

A lightweight CNN specifically designed for deepfake detection.

**Characteristics**

-   Extremely small model
-   Very fast inference
-   Suitable for embedded deployment

------------------------------------------------------------------------

## Xception

A deep CNN using depthwise separable convolutions.

**Characteristics**

-   Strong feature extraction
-   High detection accuracy
-   Widely adopted benchmark model

------------------------------------------------------------------------

## Patch ResNet

Processes local image patches independently before aggregating
predictions.

**Characteristics**

-   Learns localized forgery artifacts
-   Moderate computational cost
-   Good balance between speed and accuracy

------------------------------------------------------------------------

## Multiple Attention

A transformer-inspired architecture incorporating attention mechanisms.

**Characteristics**

-   Multi-scale feature aggregation
-   Attention-based representation learning
-   Competitive accuracy with moderate computational cost

------------------------------------------------------------------------

# Experimental Pipeline

``` text
Dataset
    │
    ▼
Face Extraction
    │
    ▼
Training
    │
    ▼
Checkpoint Selection
    │
    ▼
Test Evaluation
    │
    ├── Accuracy
    ├── Balanced Accuracy
    ├── F1 Score
    ├── MCC
    ├── ROC-AUC
    ├── PR-AUC
    ├── Confusion Matrix
    ├── FLOPs
    ├── Parameter Count
    ├── Model Size
    ├── FPS
    └── Latency
```

------------------------------------------------------------------------

# Benchmark Results

## Detection Performance

| Model | Accuracy (%) | Balanced Acc. (%) | F1 (%) | ROC-AUC | PR-AUC | MCC |
|:------|-------------:|------------------:|--------:|--------:|-------:|----:|
| Meso4 | 75.00 | 50.00 | 85.71 | 0.5368 | 0.7881 | 0.0000 |
| Xception | **82.67** | **73.56** | **88.82** | **0.8791** | **0.9531** | **0.5100** |
| Patch ResNet | 76.33 | 69.33 | 84.08 | 0.7878 | 0.9119 | 0.3801 |
| Multiple Attention | 82.00 | 72.89 | 88.36 | 0.8411 | 0.9361 | 0.4925 |

## Computational Efficiency

| Model | Parameters (M) | GFLOPs | Model Size (MB) | FPS | Latency (ms/img) |
|:------|---------------:|-------:|----------------:|----:|-----------------:|
| Meso4 | 0.03 | 0.06 | 0.3 | **6551.6** | **0.15** |
| Xception | 20.81 | 5.98 | 238.6 | 336.2 | 2.97 |
| Patch ResNet | 11.18 | 2.38 | 128.0 | 1292.6 | 0.77 |
| Multiple Attention | 12.92 | 2.48 | 148.0 | 1240.4 | 0.81 |

------------------------------------------------------------------------

## Benchmark Visualizations

### Training Curves

![Training Curves](benchmark_output/training_curves.png)

### ROC Curves

![ROC](benchmark_output/roc_curves.png)

### Precision-Recall Curves

![PR](benchmark_output/pr_curves.png)

### Confusion Matrices

![Confusion](benchmark_output/confusion_matrices.png)

### Accuracy vs Speed

![Speed](benchmark_output/accuracy_vs_fps.png)

### Model Complexity

![Parameters](benchmark_output/parameter_count.png)

------------------------------------------------------------------------

# Key Findings

-   **Xception** achieved the highest overall detection performance,
    obtaining the best ROC-AUC, PR-AUC, and MCC.
-   **Multiple Attention** delivered performance close to Xception while
    requiring significantly fewer computational resources.
-   **Patch ResNet** provided a strong compromise between inference
    speed and detection capability.
-   **Meso4** was by far the fastest and smallest model but exhibited
    substantially lower detection performance, illustrating the
    trade-off between efficiency and robustness.

------------------------------------------------------------------------

# Evaluation Metrics

Performance metrics:

-   Accuracy
-   Balanced Accuracy
-   Precision
-   Recall
-   Specificity
-   F1 Score
-   ROC-AUC
-   PR-AUC
-   Matthews Correlation Coefficient (MCC)

Efficiency metrics:

-   Parameter Count
-   FLOPs
-   Model Size
-   FPS
-   Latency

------------------------------------------------------------------------

# Running the Project

## Training

``` bash
python train.py --model xception
```

Available models:

-   meso4
-   xception
-   patch_resnet
-   multiple_attention

## Benchmarking

Open and execute:

``` text
benchmarking.ipynb
```

The notebook automatically:

-   loads trained checkpoints,
-   evaluates every model,
-   generates ROC and PR curves,
-   plots confusion matrices,
-   measures inference speed,
-   computes FLOPs and parameter counts,
-   exports benchmark tables and figures.

------------------------------------------------------------------------

## Reference

J. Deng et al., *Towards Benchmarking and Evaluating Deepfake Detection*, IEEE Transactions on Dependable and Secure Computing, 2024.

------------------------------------------------------------------------

# Team Members

-   Aditya Raj (240066)
-   Amrit Dwivedi (240111)
-   Keshav Agarwal (240537)
-   Kushagra Chandra (240585)
-   Mihir Tejaswi (240652)
