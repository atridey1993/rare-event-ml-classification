from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def sensitivity_scan(y_true, scores):
    thresholds = np.linspace(0, 1, 200)
    best = 0

    for t in thresholds:
        pred = scores > t
        S = ((y_true == 1) & pred).sum()
        B = ((y_true == 0) & pred).sum()

        val = S / np.sqrt(B + 1e-9)
        if val > best:
            best = val

    return best


def evaluate_classifier(name, y, scores):
    return {
        "name": name,
        "auc": roc_auc_score(y, scores),
        "f1": f1_score(y, scores > 0.5),
        "precision": precision_score(y, scores > 0.5),
        "recall": recall_score(y, scores > 0.5),
        "sens": sensitivity_scan(y, scores),
        "scores": scores,
        "y": y,
    }


# --------------------------
# PLOTS
# --------------------------

def plot_roc_curves(metrics, path):
    plt.figure()

    for m in metrics:
        fpr, tpr, _ = roc_curve(m["y"], m["scores"])
        plt.plot(fpr, tpr, label=f"{m['name']} (AUC={m['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_sensitivity_bars(metrics, path):
    names = [m["name"] for m in metrics]
    values = [m["sens"] for m in metrics]

    plt.bar(names, values)
    plt.ylabel("Sensitivity proxy")
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(y, scores, name, path):
    pred = (scores > 0.5).astype(int)
    cm = confusion_matrix(y, pred)

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(name)
    plt.savefig(path)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    importance = model.feature_importances_

    plt.barh(feature_names, importance)
    plt.xlabel("Importance")
    plt.savefig(path)
    plt.close()


def plot_shap(model, X, path):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(path)
    plt.close()


def save_summary(metrics, path):
    df = pd.DataFrame(metrics)
    df.to_csv(path, index=False)
