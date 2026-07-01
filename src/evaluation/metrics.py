"""Evaluation metrics for TabStream-IR experiments."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def _safe_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total == 0:
        return np.ones_like(p) / len(p)
    p = p / total
    p = np.clip(p, 0.0, None)
    p = p / p.sum()
    return p


def compute_imbalance_ratio(
    y: np.ndarray, low_q: float = 0.10, high_q: float = 0.90
) -> float:
    """IR = count(normal) / count(rare) where rare = bottom/top quantile."""
    y = np.asarray(y).ravel()
    lo = np.quantile(y, low_q)
    hi = np.quantile(y, high_q)
    n_rare = np.sum((y <= lo) | (y >= hi))
    n_normal = len(y) - n_rare
    return n_normal / max(n_rare, 1)


def tail_thresholds(
    y: np.ndarray, low_q: float = 0.10, high_q: float = 0.90
) -> tuple[float, float]:
    """Return fixed low/high target thresholds for tail evaluation."""
    y = np.asarray(y).ravel()
    return float(np.quantile(y, low_q)), float(np.quantile(y, high_q))


def fixed_tail_mask(y: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    """Rare-region mask using precomputed thresholds."""
    y = np.asarray(y).ravel()
    lo, hi = thresholds
    return (y <= lo) | (y >= hi)


def compute_imbalance_ratio_fixed(
    y: np.ndarray, thresholds: tuple[float, float]
) -> float:
    """IR using fixed rare-region thresholds."""
    rare = fixed_tail_mask(y, thresholds)
    n_rare = int(rare.sum())
    n_normal = len(rare) - n_rare
    return n_normal / max(n_rare, 1)


def mse_rare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> float:
    """MSE restricted to the rare (tail) region of the target."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = np.quantile(y_true, low_q)
    hi = np.quantile(y_true, high_q)
    mask = (y_true <= lo) | (y_true >= hi)
    if mask.sum() == 0:
        return 0.0
    return float(mean_squared_error(y_true[mask], y_pred[mask]))


def mse_rare_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, float],
) -> float:
    """MSE restricted to a fixed rare region."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = fixed_tail_mask(y_true, thresholds)
    if mask.sum() == 0:
        return 0.0
    return float(mean_squared_error(y_true[mask], y_pred[mask]))


def mae_rare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> float:
    """MAE restricted to the rare region."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = np.quantile(y_true, low_q)
    hi = np.quantile(y_true, high_q)
    mask = (y_true <= lo) | (y_true >= hi)
    if mask.sum() == 0:
        return 0.0
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def mae_rare_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, float],
) -> float:
    """MAE restricted to a fixed rare region."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = fixed_tail_mask(y_true, thresholds)
    if mask.sum() == 0:
        return 0.0
    return float(mean_absolute_error(y_true[mask], y_pred[mask]))


def f1_rare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low_q: float = 0.10,
    high_q: float = 0.90,
    tolerance: float = 0.5,
) -> float:
    """F1-like score for correctly predicting rare-region membership.

    A prediction is 'correct' if it is within ``tolerance * std(y_true)``
    of the true value for a rare-region sample.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    lo = np.quantile(y_true, low_q)
    hi = np.quantile(y_true, high_q)
    rare = (y_true <= lo) | (y_true >= hi)
    n_rare = rare.sum()
    if n_rare == 0:
        return 1.0

    thresh = tolerance * np.std(y_true)
    correct = np.abs(y_true - y_pred) <= thresh

    tp = np.sum(rare & correct)
    fn = np.sum(rare & ~correct)
    pred_rare = (y_pred <= lo) | (y_pred >= hi)
    fp = np.sum(pred_rare & ~rare)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def f1_rare_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: tuple[float, float],
    tolerance: float = 0.5,
) -> float:
    """F1-like rare score using fixed rare-region thresholds."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    rare = fixed_tail_mask(y_true, thresholds)
    n_rare = rare.sum()
    if n_rare == 0:
        return 1.0

    thresh = tolerance * np.std(y_true)
    correct = np.abs(y_true - y_pred) <= thresh

    tp = np.sum(rare & correct)
    fn = np.sum(rare & ~correct)
    pred_rare = fixed_tail_mask(y_pred, thresholds)
    fp = np.sum(pred_rare & ~rare)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def throughput_records_per_sec(
    n_records: int, elapsed_seconds: float
) -> float:
    return n_records / max(elapsed_seconds, 1e-12)


def drift_detection_delay(
    true_drift_indices: list[int],
    detected_drift_indices: list[int],
    tolerance: int = 10,
) -> dict:
    """Measure detection delay and missed/false drift events.

    Returns dict with: mean_delay, n_detected, n_missed, n_false.
    """
    delays: list[int] = []
    detected_set = set(detected_drift_indices)
    matched_detected: set[int] = set()

    for true_idx in true_drift_indices:
        best_delay: Optional[int] = None
        for det_idx in detected_drift_indices:
            d = det_idx - true_idx
            if 0 <= d <= tolerance * 10:
                if best_delay is None or d < best_delay:
                    best_delay = d
                    matched_detected.add(det_idx)
        if best_delay is not None:
            delays.append(best_delay)

    n_detected = len(delays)
    n_missed = len(true_drift_indices) - n_detected
    n_false = len(detected_set - matched_detected)
    mean_delay = float(np.mean(delays)) if delays else float("inf")

    return {
        "mean_delay": mean_delay,
        "n_detected": n_detected,
        "n_missed": n_missed,
        "n_false": n_false,
    }


def evaluate_downstream(
    y_train: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    X_test: np.ndarray,
    model_type: str = "xgboost",
    n_estimators: int = 100,
    seed: int = 42,
    sample_weight: Optional[np.ndarray] = None,
    rare_thresholds: Optional[tuple[float, float]] = None,
) -> dict:
    """Train a downstream regressor and compute metrics."""
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=n_estimators,
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, random_state=seed
            )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=n_estimators,
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, random_state=seed
            )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=n_estimators, random_state=seed
        )

    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=np.asarray(sample_weight).ravel())
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if rare_thresholds is None:
        rare_mse = mse_rare(y_test, y_pred)
        rare_mae = mae_rare(y_test, y_pred)
        rare_f1 = f1_rare(y_test, y_pred)
    else:
        rare_mse = mse_rare_fixed(y_test, y_pred, rare_thresholds)
        rare_mae = mae_rare_fixed(y_test, y_pred, rare_thresholds)
        rare_f1 = f1_rare_fixed(y_test, y_pred, rare_thresholds)

    return {
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse_rare": rare_mse,
        "mae_rare": rare_mae,
        "f1_rare": rare_f1,
    }
