from pathlib import Path

from src.data_generation import generate_tabular_dataset, generate_image_dataset
from src.preprocess import prepare_tabular_data, prepare_image_data
from src.train_models import (
    run_cut_based_baseline,
    train_xgboost_model,
    train_ann_model,
    train_cnn_model,
)
from src.evaluate import (
    evaluate_classifier,
    plot_roc_curves,
    plot_sensitivity_bars,
    save_summary_table,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_shap,
)
from src.utils import ensure_dir


def main() -> None:
    results_dir = Path("results")
    ensure_dir(results_dir)

    # -----------------------------
    # Tabular data for baseline / XGBoost / ANN
    # -----------------------------
    df = generate_tabular_dataset(
        n_signal=15000,
        n_background=120000,
        random_state=42,
    )

    feature_columns = [
        "pt_lead",
        "pt_sublead",
        "met",
        "mjj",
        "delta_r",
        "ht",
        "jet_mass",
        "tau21",
        "ecf_ratio",
        "central_energy_fraction",
    ]

    tabular_data = prepare_tabular_data(
        df=df,
        feature_columns=feature_columns,
        target_column="label",
    )

    baseline = run_cut_based_baseline(
        tabular_data["X_test_raw"],
        tabular_data["y_test"],
    )

    xgb_model = train_xgboost_model(tabular_data)
    ann_model = train_ann_model(tabular_data)

    baseline_scores = baseline["scores"]
    xgb_scores = xgb_model.predict_proba(tabular_data["X_test_scaled"])[:, 1]
    ann_scores = ann_model.predict(
        tabular_data["X_test_scaled"],
        verbose=0,
    ).ravel()

    baseline_metrics = evaluate_classifier(
        model_name="Cut-based",
        y_true=tabular_data["y_test"],
        scores=baseline_scores,
    )

    xgb_metrics = evaluate_classifier(
        model_name="XGBoost",
        y_true=tabular_data["y_test"],
        scores=xgb_scores,
    )

    ann_metrics = evaluate_classifier(
        model_name="ANN",
        y_true=tabular_data["y_test"],
        scores=ann_scores,
    )

    # -----------------------------
    # Image-like data for CNN
    # -----------------------------
    X_img, y_img = generate_image_dataset(
        n_signal=8000,
        n_background=32000,
        image_shape=(16, 16),
        random_state=42,
    )

    image_data = prepare_image_data(X_img, y_img)
    cnn_model = train_cnn_model(image_data)

    cnn_scores = cnn_model.predict(
        image_data["X_test"],
        verbose=0,
    ).ravel()

    cnn_metrics = evaluate_classifier(
        model_name="CNN",
        y_true=image_data["y_test"],
        scores=cnn_scores,
    )

    # -----------------------------
    # Collect all metrics
    # -----------------------------
    all_metrics = [
        baseline_metrics,
        xgb_metrics,
        ann_metrics,
        cnn_metrics,
    ]

    # -----------------------------
    # Save comparison plots / tables
    # -----------------------------
    plot_roc_curves(
        metrics_list=all_metrics,
        output_path=results_dir / "roc_curves.png",
    )

    plot_sensitivity_bars(
        metrics_list=all_metrics,
        output_path=results_dir / "sensitivity_bars.png",
    )

    save_summary_table(
        metrics_list=all_metrics,
        output_path=results_dir / "summary_metrics.csv",
    )

    # -----------------------------
    # Diagnostics for tabular models
    # -----------------------------
    plot_confusion_matrix(
        y_true=tabular_data["y_test"],
        scores=xgb_scores,
        model_name="XGBoost",
        output_path=results_dir / "cm_xgb.png",
    )

    plot_confusion_matrix(
        y_true=tabular_data["y_test"],
        scores=ann_scores,
        model_name="ANN",
        output_path=results_dir / "cm_ann.png",
    )

    plot_feature_importance(
        model=xgb_model,
        feature_names=feature_columns,
        output_path=results_dir / "feature_importance.png",
    )

    # Use raw tabular test features for readable SHAP labels if your evaluate.py supports it
    plot_shap(
        model=xgb_model,
        X=tabular_data["X_test_scaled"][:1000],
        output_path=results_dir / "shap.png",
    )

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n=== Summary ===")
    for m in all_metrics:
        print(
            f"{m['model_name']}: "
            f"ROC-AUC={m['roc_auc']:.4f}, "
            f"Precision={m['precision']:.4f}, "
            f"Recall={m['recall']:.4f}, "
            f"F1={m['f1']:.4f}, "
            f"SensitivityProxy={m['best_sensitivity_proxy']:.4f}"
        )

    print("\nSaved outputs:")
    print(f"- {results_dir / 'roc_curves.png'}")
    print(f"- {results_dir / 'sensitivity_bars.png'}")
    print(f"- {results_dir / 'summary_metrics.csv'}")
    print(f"- {results_dir / 'cm_xgb.png'}")
    print(f"- {results_dir / 'cm_ann.png'}")
    print(f"- {results_dir / 'feature_importance.png'}")
    print(f"- {results_dir / 'shap.png'}")


if __name__ == "__main__":
    main()
