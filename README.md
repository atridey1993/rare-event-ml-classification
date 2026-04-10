# Rare-Event ML Classification in Noisy, Imbalanced Data

## Overview
This project demonstrates an end-to-end machine learning pipeline for detecting rare signals in noisy, highly imbalanced datasets. It is inspired by workflows where synthetic events are generated, signal and background classes are simulated, simple rule-based selections are benchmarked, and machine learning models are used to improve classification performance.

The goal is to show practical skills in:
- simulation-driven analytics
- feature engineering
- imbalanced classification
- model comparison
- evaluation with both ML metrics and significance-style measures

## Motivation
Many real-world classification tasks involve subtle signals hidden inside dominant background noise. This project reproduces that setting with synthetic collider-inspired observables and compares:
- a baseline cut-based strategy
- XGBoost
- a neural network classifier

## Methods
- Synthetic rare-event data generation
- Train/validation/test split
- Feature scaling
- Baseline cut-based selection
- XGBoost classifier
- MLP neural network classifier
- Evaluation with ROC-AUC, F1-score, precision, recall
- Significance-style scan using \( S/\sqrt{B} \)

## Repository Structure
- `src/data_generation.py` : synthetic event generation
- `src/preprocess.py` : train/test split and scaling
- `src/train_models.py` : baseline + ML models
- `src/evaluate.py` : metrics, significance scan, plots
- `main.py` : runs the full pipeline

## Example Features
The synthetic dataset includes collider-inspired variables such as:
- transverse momentum-like observables
- missing-energy-like variable
- invariant-mass-like variable
- angular separation
- event activity
- jet substructure-inspired features
- detector-response-inspired energy fraction

## Skills Demonstrated
- Python
- machine learning
- feature engineering
- imbalanced classification
- evaluation and validation
- simulation-driven data science
- scientific and statistical reasoning

## Results
The ML models are expected to outperform the simple baseline selection in identifying rare signals under strong class imbalance. The project also shows how to compare models using both standard ML metrics and a significance-style metric relevant for rare-event detection.

## Future Extensions
- hyperparameter optimization
- SHAP-based explainability
- calibration analysis
- Keras/TensorFlow deep neural network
- dashboard or lightweight app for model inspection
