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
)
from src.utils import ensure_dir


def main() -> None:
    results_dir = Path("results")
    ensure_dir(results_dir)

    # -----------------------------
    # Tabular data for XGBoost / ANN
    # -----------------------------
    df = generate_tabular_dataset(n_signal=15000, n_background=120000)

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

    tabular_data = prepare_tabular_data(df, feature_columns)

    baseline = run_cut_based_baseline(
        tabular_data["X_test_raw"],
        tabular_data["y_test"],
    )

    xgb_model = train_xgboost_model(tabular_data)
    ann_model = train_ann_model(tabular_data)

    baseline_metrics = evaluate_classifier(
        "Cut-based",
        tabular_data["y_test"],
        baseline["scores"],
    )

    xgb_metrics = evaluate_classifier(
        "XGBoost",
        tabular_data["y_test"],
        xgb_model.predict_proba(tabular_data["X_test_scaled"])[:, 1],
    )

    ann_metrics = evaluate_classifier(
        "ANN",
        tabular_data["y_test"],
        ann_model.predict(tabular_data["X_test_scaled"], verbose=0).ravel(),
    )

    # -----------------------------
    # Image-like data for CNN
    # -----------------------------
    X_img, y_img = generate_image_dataset(
        n_signal=8000,
        n_background=32000,
        image_shape=(16, 16),
    )

    image_data = prepare_image_data(X_img, y_img)
    cnn_model = train_cnn_model(image_data)

    cnn_metrics = evaluate_classifier(
        "CNN",
        image_data["y_test"],
        cnn_model.predict(image_data["X_test"], verbose=0).ravel(),
    )

    all_metrics = [baseline_metrics, xgb_metrics, ann_metrics, cnn_metrics]

    plot_roc_curves(all_metrics, results_dir / "roc_curves.png")
    plot_sensitivity_bars(all_metrics, results_dir / "sensitivity_bars.png")
    save_summary_table(all_metrics, results_dir / "summary_metrics.csv")

    print("\n=== Summary ===")
    for m in all_metrics:
        print(
            f"{m['model_name']}: "
            f"ROC-AUC={m['roc_auc']:.4f}, "
            f"F1={m['f1']:.4f}, "
            f"SensitivityProxy={m['best_sensitivity_proxy']:.4f}"
        )

    print("\nSaved:")
    print(results_dir / "roc_curves.png")
    print(results_dir / "sensitivity_bars.png")
    print(results_dir / "summary_metrics.csv")


if __name__ == "__main__":
    main()
