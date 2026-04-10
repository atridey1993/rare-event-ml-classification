# Rare-Event Detection with Machine Learning in Noisy, Imbalanced Data

## Overview
This project implements an end-to-end machine learning pipeline for detecting rare signals in noisy and highly imbalanced datasets. It is inspired by simulation-driven workflows where signal and background events are generated, baseline selections are applied, and machine learning models are used to improve classification performance.

The goal is to demonstrate practical skills in:
- simulation-driven data analysis
- feature engineering
- imbalanced classification
- model comparison
- domain-aware evaluation

---

## Motivation
Many real-world problems involve detecting weak signals embedded in large background noise. This project reproduces that setting and compares:

- a baseline cut-based selection
- XGBoost classifier
- Neural network classifier

---

## Methods

### Data
Synthetic dataset with signal vs background classes using collider-inspired observables:
- transverse momentum-like variables
- missing-energy-like variable
- invariant-mass-like variable
- angular separation
- event activity and jet-inspired features

### Models
- Baseline: rule-based cut selection
- XGBoost
- Neural Network (MLP)

### Evaluation
- ROC-AUC
- Precision, Recall, F1-score
- Threshold-based sensitivity proxy inspired by signal/background analysis

---

## Sensitivity Proxy (Important Note)
This project includes a simple threshold-based sensitivity proxy using signal and background yields for illustration purposes.

This is **not a full statistical significance calculation**. In realistic analyses, discovery sensitivity depends on:
- likelihood-based statistical treatment
- systematic uncertainties
- full event modeling

Here, the proxy is used only to provide an intuitive comparison between models in a rare-event setting.

---

## Results
Machine learning models (XGBoost and neural network) outperform the baseline cut-based method in detecting rare signals in noisy, imbalanced data. The sensitivity proxy provides an additional comparison of model performance alongside standard ML metrics.

---

## Repository Structure
