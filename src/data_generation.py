from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_positive(values: np.ndarray, minimum: float = 0.0) -> np.ndarray:
    return np.clip(values, minimum, None)


def generate_tabular_dataset(
    n_signal: int = 10000,
    n_background: int = 100000,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    signal = pd.DataFrame(
        {
            "pt_lead": _clip_positive(rng.normal(260, 55, n_signal)),
            "pt_sublead": _clip_positive(rng.normal(180, 40, n_signal)),
            "met": _clip_positive(rng.normal(220, 60, n_signal)),
            "mjj": _clip_positive(rng.normal(125, 18, n_signal)),
            "delta_r": _clip_positive(rng.normal(1.1, 0.35, n_signal)),
            "ht": _clip_positive(rng.normal(620, 110, n_signal)),
            "jet_mass": _clip_positive(rng.normal(95, 18, n_signal)),
            "tau21": np.clip(rng.normal(0.35, 0.10, n_signal), 0.0, 1.0),
            "ecf_ratio": np.clip(rng.normal(0.65, 0.12, n_signal), 0.0, 2.0),
            "central_energy_fraction": np.clip(
                rng.normal(0.72, 0.08, n_signal), 0.0, 1.0
            ),
            "label": np.ones(n_signal, dtype=int),
        }
    )

    background = pd.DataFrame(
        {
            "pt_lead": _clip_positive(rng.normal(180, 65, n_background)),
            "pt_sublead": _clip_positive(rng.normal(110, 45, n_background)),
            "met": _clip_positive(rng.normal(120, 70, n_background)),
            "mjj": _clip_positive(rng.normal(90, 35, n_background)),
            "delta_r": _clip_positive(rng.normal(1.8, 0.55, n_background)),
            "ht": _clip_positive(rng.normal(430, 140, n_background)),
            "jet_mass": _clip_positive(rng.normal(70, 25, n_background)),
            "tau21": np.clip(rng.normal(0.58, 0.14, n_background), 0.0, 1.0),
            "ecf_ratio": np.clip(rng.normal(0.42, 0.16, n_background), 0.0, 2.0),
            "central_energy_fraction": np.clip(
                rng.normal(0.52, 0.12, n_background), 0.0, 1.0
            ),
            "label": np.zeros(n_background, dtype=int),
        }
    )

    df = pd.concat([signal, background], ignore_index=True)
    df["ht"] = df["ht"] + 0.3 * df["pt_lead"] + 0.2 * df["pt_sublead"]
    df["mjj"] = df["mjj"] + 0.02 * df["jet_mass"] * (1.0 - df["tau21"])
    df["met"] = df["met"] + 20.0 * df["central_energy_fraction"]
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def generate_image_dataset(
    n_signal: int = 5000,
    n_background: int = 20000,
    image_shape: tuple[int, int] = (16, 16),
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    h, w = image_shape

    def make_signal_image() -> np.ndarray:
        img = rng.normal(0.05, 0.03, size=(h, w))
        cx, cy = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                dist2 = (i - cx) ** 2 + (j - cy) ** 2
                img[i, j] += 0.8 * np.exp(-dist2 / 18.0)
        return np.clip(img, 0.0, None)

    def make_background_image() -> np.ndarray:
        img = rng.normal(0.08, 0.04, size=(h, w))
        stripe_col = rng.integers(0, w)
        img[:, stripe_col] += rng.normal(0.08, 0.03, size=h)
        return np.clip(img, 0.0, None)

    signal = np.stack([make_signal_image() for _ in range(n_signal)], axis=0)
    background = np.stack([make_background_image() for _ in range(n_background)], axis=0)

    X = np.concatenate([signal, background], axis=0)
    y = np.concatenate([np.ones(n_signal, dtype=int), np.zeros(n_background, dtype=int)])

    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]

    # Add channel dimension for CNN
    X = X[..., np.newaxis].astype(np.float32)
    return X, y
