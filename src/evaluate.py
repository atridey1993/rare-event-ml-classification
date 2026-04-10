from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def sensitivity_scan(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 300)

    best_threshold = 0.5
    best_proxy = -1.0
    proxy_values = []

    for thr in thresholds:
        pred = scores >= thr
        s = int(((y_true == 1) & pred).sum())
        b = int(((y_true == 0) & pred).sum())

        # Simple proxy for illustrative comparison only
        proxy = s / np.sqrt(b + 1e-9) if b > 0 else 0.0
        proxy_values.append(proxy)

        if proxy > best_proxy:
            best_proxy = proxy
            best_threshold = thr

    return {
        "thresholds": thresholds,
        "proxy_values": np.array(proxy_values),
        "best_threshold": best_threshold,
        "best_sensitivity_proxy": best_proxy,
    }


def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:
    roc_auc = roc_auc_score(y_true, scores)

    scan = sensitivity_scan(y_true, scores)
    best_thr = scan["best_threshold"]
    y_pred = (scores >= best_thr).astype(int)

    return {
        "model_name": model_name,
        "scores": scores,
        "y_true": y_true,
        "roc_auc": roc_auc,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "best_threshold": best_thr,
        "best_sensitivity_proxy": scan["best_sensitivity_proxy"],
        "scan": scan,
    }


def plot_roc_curves(metrics_list: list[dict], output_path: Path) -> None:
    plt.figure(figsize=(7, 5))

    for metrics in metrics_list:
        fpr, tpr, _ = roc_curve(metrics["y_true"], metrics["scores"])
        plt.plot(
            fpr,
            tpr,
            label=f"{metrics['model_name']} (AUC={metrics['roc_auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sensitivity_bars(metrics_list: list[dict], output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    names = [m["model_name"] for m in metrics_list]
    values = [m["best_sensitivity_proxy"] for m in metrics_list]

    plt.bar(names, values)
    plt.ylabel("Best sensitivity proxy")
    plt.title("Threshold-based Sensitivity Comparison")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list[str],
    output_path: Path,
) -> None:
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model does not provide feature_importances_.")

    importances = model.feature_importances_
    order = np.argsort(importances)

    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[order], importances[order])
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_shap(
    model,
    X: np.ndarray,
    output_path: Path,
) -> None:
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_summary_table(metrics_list: list[dict], output_path: Path) -> None:
    rows = []
    for metrics in metrics_list:
        rows.append(
            {
                "model_name": metrics["model_name"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "best_threshold": metrics["best_threshold"],
                "best_sensitivity_proxy": metrics["best_sensitivity_proxy"],
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)
