from __future__ import annotations

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier


def run_cut_based_baseline(X_test_raw: pd.DataFrame, y_test: np.ndarray) -> dict:
    score = (
        0.018 * X_test_raw["pt_lead"]
        + 0.015 * X_test_raw["met"]
        + 0.010 * X_test_raw["ht"]
        + 0.020 * X_test_raw["mjj"]
        - 45.0 * X_test_raw["tau21"]
        - 8.0 * X_test_raw["delta_r"]
        + 12.0 * X_test_raw["ecf_ratio"]
        + 10.0 * X_test_raw["central_energy_fraction"]
    )

    score_min = score.min()
    score_max = score.max()
    norm_score = (score - score_min) / (score_max - score_min + 1e-12)

    return {
        "scores": np.asarray(norm_score),
        "y_test": y_test,
    }


def train_xgboost_model(data: dict) -> XGBClassifier:
    y_train = data["y_train"]
    scale_pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0,
    )

    model.fit(
        data["X_train_scaled"],
        data["y_train"],
        eval_set=[(data["X_val_scaled"], data["y_val"])],
        verbose=False,
    )
    return model


def train_ann_model(data: dict) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(data["X_train_scaled"].shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        data["X_train_scaled"],
        data["y_train"],
        validation_data=(data["X_val_scaled"], data["y_val"]),
        epochs=30,
        batch_size=256,
        verbose=0,
        callbacks=[early_stopping],
    )
    return model


def train_cnn_model(data: dict) -> keras.Model:
    input_shape = data["X_train"].shape[1:]

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=20,
        batch_size=256,
        verbose=0,
        callbacks=[early_stopping],
    )
    return model
