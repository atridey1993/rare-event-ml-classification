# Rare-Event Detection with XGBoost, ANN, and CNN in Noisy, Imbalanced Data

## Overview
This project implements an end-to-end machine learning pipeline for detecting rare signals in noisy and highly imbalanced datasets. It is inspired by simulation-driven workflows where signal and background events are generated, baseline selections are applied, and machine learning models are used to improve classification performance.

The project demonstrates practical skills in:
- simulation-driven data analysis
- feature engineering
- imbalanced classification
- model comparison
- domain-aware evaluation

## Motivation
Many real-world problems involve detecting weak signals amid large amounts of background noise. This project reproduces that setting and compares:

- a baseline cut-based selection
- XGBoost on tabular event-level features
- ANN on tabular event-level features
- CNN on detector-inspired image-like inputs

## Methods

### Data
Synthetic dataset with signal vs background classes using collider-inspired observables:
- transverse momentum-like variables
- missing-energy-like variable
- invariant-mass-like variable
- angular separation
- event activity and jet-inspired features

In addition, a simple detector-inspired 2D image representation is generated for CNN-based classification.

### Models
- Baseline: rule-based cut selection
- XGBoost
- ANN (dense neural network on tabular features)
- CNN (convolutional neural network on 2D image-like inputs)

### Evaluation
- ROC-AUC
- Precision, Recall, F1-score
- Threshold-based sensitivity proxy inspired by signal/background analysis

## Sensitivity Proxy (Important Note)
This project includes a simple threshold-based sensitivity proxy using signal and background yields for illustration purposes.

This is not a full statistical significance calculation. In realistic analyses, discovery sensitivity depends on:
- likelihood-based statistical treatment
- systematic uncertainties
- full event modeling

Here, the proxy is used only to provide an intuitive comparison between models in a rare-event setting.

## Results
The project compares a rule-based baseline with XGBoost, ANN, and CNN classifiers. The ML models are expected to outperform the baseline in rare-event detection under noisy and imbalanced conditions.

## Repository Structure
- `src/data_generation.py` : synthetic tabular and image-like data generation
- `src/preprocess.py` : preprocessing and train/test split
- `src/train_models.py` : baseline, XGBoost, ANN, CNN
- `src/evaluate.py` : evaluation, plots, sensitivity proxy
- `main.py` : runs the full pipeline

## Skills Demonstrated
- Python
- XGBoost
- Artificial Neural Networks
- Convolutional Neural Networks
- Feature engineering
- Imbalanced classification
- Simulation-driven analytics
- Statistical reasoning

## Future Improvements
- hyperparameter tuning
- explainability (SHAP)
- calibration analysis
- detector-response realism
- dashboard deployment
