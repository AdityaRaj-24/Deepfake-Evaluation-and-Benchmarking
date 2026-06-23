# Benchmarking Deepfake Detection Methods

As a course project under EE656, we present the implementation of the paper:

**Towards Benchmarking and Evaluating Deepfake Detection**  
Jingyi Deng et al., IEEE Transactions on Dependable and Secure Computing, 2024

---

## Overview

Deepfake generation techniques have advanced rapidly in recent years, creating highly realistic synthetic media that poses significant challenges in misinformation detection, digital forensics, privacy protection, and cybersecurity.

This project presents a comparative benchmarking study of representative deepfake detection architectures inspired by the evaluation framework proposed in the paper.  

Rather than reproducing the complete benchmark containing thirteen detection algorithms, this project implements and evaluates four representative detectors spanning multiple deepfake detection paradigms:

* Meso4 — Lightweight CNN-based detector
* Xception — Deep CNN baseline
* Patch ResNet Layer1 — Patch-level artifact detector
* M2TR — Multi-modal Transformer-based detector

The goal is to compare their effectiveness, efficiency, robustness, and deployment practicality under a unified evaluation framework.

## Project Objectives

The project aims to:

* Implement representative deepfake detection architectures.
* Compare multiple detection philosophies under identical evaluation conditions.
* Analyze performance beyond accuracy using practical deployment metrics.
* Study the trade-off between computational cost and detection capability.
* Investigate robustness under image perturbations and compression artifacts.

## Detection Architectures

### 1. Meso4

Meso4 is a lightweight convolutional neural network designed specifically for deepfake detection.

Characteristics:

* Shallow architecture
* Low parameter count
* Fast inference
* Suitable for resource-constrained environments

Detection philosophy:

```text
Face Image
     ↓
Mesoscopic Feature Extraction
     ↓
Binary Classification
```

---

### 2. Xception

Xception serves as one of the most widely adopted deepfake detection baselines.

Characteristics:

* Deep separable convolutions
* Strong feature extraction capability
* High benchmark popularity
* Robust image-level classification

Detection philosophy:

```text
Face Image
     ↓
XceptionNet
     ↓
Real / Fake
```

---

### 3. Patch ResNet Layer1

Patch-based detection focuses on local manipulation artifacts instead of global image appearance.

Characteristics:

* Local artifact learning
* Patch-level supervision
* Better sensitivity to small forgery regions

Detection philosophy:

```text
Image
  ↓
Patch Extraction
  ↓
Patch Classification
  ↓
Prediction Aggregation
  ↓
Final Decision
```

---

### 4. M2TR

M2TR (Multi-modal Multi-scale Transformer) represents a modern state-of-the-art deepfake detector.

Characteristics:

* Transformer-based architecture
* Spatial feature learning
* Frequency-domain analysis
* Multi-stream feature fusion

Detection philosophy:

```text
Image
   ├── Spatial Stream
   ├── Frequency Stream
   ↓
Cross-Modal Fusion
   ↓
Transformer Classification
```

---

## Methodology

### Data Preprocessing

The preprocessing pipeline consists of:

1. Video frame extraction
2. Face detection
3. Face alignment
4. Face cropping
5. Dataset generation

Workflow:

```text
Input Video
      ↓
Frame Extraction
      ↓
Face Detection
      ↓
Face Alignment
      ↓
Face Cropping
      ↓
Training Dataset
```

---

## Repository Structure

```text
.
├── classifiers.py
├── pipeline.py
├── train.ipynb
├── benchmarking.ipynb
├── models/
│   ├── meso4/
│   ├── xception/
│   ├── patch_resnet/
│   └── m2tr/
│
├── train_faces/
├── test_faces/
├── weights/
└── README.md
```

---

## Evaluation Metrics

The benchmark evaluates models using multiple complementary metrics.

### Detection Performance

* Accuracy
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC-AUC

### Robustness Analysis

* Contrast Adjustment
* Saturation Adjustment
* Gaussian Blur
* JPEG Compression
* Gaussian Noise
  
### Model Complexity
* FLOPs
* Parameter Count

### Efficiency

* Inference Time
* Throughput

---

## Experimental Workflow

```text
Dataset
   ↓
Preprocessing
   ↓

Meso4
Xception
Patch ResNet
M2TR

   ↓

Prediction Generation
   ↓

Accuracy
F1
ROC-AUC
Confusion Matrix
FLOPs
Parameters
Inference Time
Robustness Analysis
```

---

## Expected Comparative Analysis

The benchmark is designed to answer the following questions:

* How much performance is gained by moving from lightweight CNNs to deep CNNs?
* Do patch-based methods improve local forgery detection?
* Does frequency-domain information improve generalization?
* Is the additional complexity of transformer-based detectors justified?
* Which model offers the best balance between accuracy and efficiency?

---

## Results

### Detection Performance

| Model        | Accuracy | F1 Score | AUC |
| ------------ | -------- | -------- | --- |
| Meso4        | TBD      | TBD      | TBD |
| Xception     | TBD      | TBD      | TBD |
| Patch ResNet | TBD      | TBD      | TBD |
| M2TR         | TBD      | TBD      | TBD |

### Efficiency Metrics

| Model        | Parameters | FLOPs | Inference Time |
| ------------ | ---------- | ----- | -------------- |
| Meso4        | TBD        | TBD   | TBD            |
| Xception     | TBD        | TBD   | TBD            |
| Patch ResNet | TBD        | TBD   | TBD            |
| M2TR         | TBD        | TBD   | TBD            |

---

## Key Contributions

* Implementation of four representative deepfake detection paradigms.
* Unified benchmarking framework for fair comparison.
* Robustness evaluation under realistic perturbations.
* Computational efficiency analysis.
* Practical comparison between CNN-based, patch-based, and transformer-based approaches.

---

## Limitations

This project does not attempt to reproduce the full benchmark proposed by Deng et al. (2024), which includes thirteen detection algorithms and a large-scale Imperceptible and Diverse (ID) Test Set.

Instead, the focus is on a carefully selected subset of representative architectures that provide meaningful insight into modern deepfake detection strategies while remaining feasible within the scope of a course project.

---

## References

J. Deng, C. Lin, P. Hu, C. Shen, Q. Wang, Q. Li and Q. Li,

"Towards Benchmarking and Evaluating Deepfake Detection,"

IEEE Transactions on Dependable and Secure Computing, 2024.

---

## Team Members:

* Aditya Raj (240066)
* Amrit Dwivedi (240111)
* Keshav Agarwal (240537)
* Kushagra Chandra (240585)
* Mihir Tejaswi (240652)

---
