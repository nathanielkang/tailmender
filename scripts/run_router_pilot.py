#!/usr/bin/env python3
"""Parity-first pilot for the TabStream-IR sketch-weighted tail router.

This is the active E2 experiment entry point. It intentionally avoids the
quarantined copula pipeline and writes machine-readable pilot results only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    compute_imbalance_ratio_fixed,
    evaluate_downstream,
    fixed_tail_mask,
    tail_thresholds,
)


PILOT_DATASETS = (
    "tail_shift_linear",
    "low_tail_nonlinear",
    "heteroscedastic_tail",
)

REAL_WORLD_DATASETS = (
    "diabetes",
    "california_housing",
    "wine_quality",
    "airfoil",
    "abalone",
)

UAI9_DATASETS = (
    "abalone",
    "cpu_activity",
    "california_housing",
    "house_16h",
    "insurance",
    "ames_housing",
    "ibm_employee",
    "melbourne_housing",
    "bike_sharing",
)


@dataclass
class RouterConfig:
    batch_size: int = 64
    window_size: int = 384
    tail_weight: float = 5.0
    deficit_weight: float = 4.0
    drift_weight: float = 1.5
    replay_ratio: float = 0.35
    max_weight: float = 14.0
    low_q: float = 0.10
    high_q: float = 0.90


@dataclass
class RouterStats:
    total_real: int = 0
    total_replay: int = 0
    mean_tail_mass: float = 0.0
    mean_deficit: float = 0.0
    mean_weight: float = 1.0
    max_weight: float = 1.0
    n_batches: int = 0


def make_pilot_dataset(name: str, seed: int, n: int = 1800, d: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Create explicit synthetic stress streams for pilot-only testing."""
    if name not in PILOT_DATASETS:
        raise ValueError(f"Unknown pilot dataset '{name}'. Choose from {PILOT_DATASETS}.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    X = rng.normal(0.0, 1.0, size=(n, d))

    # Chronological covariate drift: later windows carry more tail exposure.
    X[:, 0] += np.where(t > 0.55, 0.9 + 0.6 * (t - 0.55), 0.0)
    X[:, 1] += np.where(t > 0.70, -0.7, 0.0)

    if name == "tail_shift_linear":
        rare_signal = np.maximum(X[:, 0] - 1.15, 0.0) ** 2
        y = 2.2 * X[:, 0] + 1.3 * X[:, 2] + 7.0 * rare_signal
        y += rng.normal(0.0, 0.7, n)
    elif name == "low_tail_nonlinear":
        rare_signal = np.maximum(-X[:, 1] - 1.05, 0.0) ** 2
        y = 1.5 * np.sin(X[:, 0]) - 1.8 * X[:, 1] - 7.5 * rare_signal
        y += rng.normal(0.0, 0.6, n)
    else:
        upper = np.maximum(X[:, 0] - 1.0, 0.0)
        lower = np.maximum(-X[:, 1] - 1.0, 0.0)
        noise_scale = 0.35 + 1.25 * (upper + lower > 0)
        y = X[:, 0] - 1.2 * X[:, 1] + 5.5 * upper**2 - 5.0 * lower**2
        y += rng.normal(0.0, noise_scale, n)

    return X.astype(np.float64), y.astype(np.float64)


def frame_to_arrays(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert a mixed-type tabular regression frame to numeric arrays."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    target_col = target_col.strip()
    clean = df.dropna(subset=[target_col]).copy()
    y = pd.to_numeric(clean[target_col], errors="coerce")
    clean = clean.loc[y.notna()].copy()
    y = y.loc[y.notna()].to_numpy(dtype=np.float64)
    X_df = clean.drop(columns=[target_col])
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    keep_cols = [col for col in X_df.columns if X_df[col].notna().mean() >= 0.60]
    X_df = X_df[keep_cols]
    num_cols = X_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [col for col in X_df.columns if col not in num_cols]
    if num_cols:
        X_num = X_df[num_cols].apply(pd.to_numeric, errors="coerce")
        X_num = X_num.fillna(X_num.median(numeric_only=True))
    else:
        X_num = pd.DataFrame(index=X_df.index)
    if cat_cols:
        X_cat = X_df[cat_cols].astype(str).replace({"nan": "missing", "None": "missing"})
        X_cat = X_cat.apply(lambda col: pd.factorize(col, sort=True)[0])
    else:
        X_cat = pd.DataFrame(index=X_df.index)
    X_out = pd.concat([X_num, X_cat], axis=1)
    return X_out.to_numpy(dtype=np.float64), y


def openml_to_arrays(
    dataset_name: str,
    target_col: str | None = None,
    version: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if version is None:
        data = fetch_openml(dataset_name, as_frame=True, parser="auto")
    else:
        data = fetch_openml(dataset_name, version=version, as_frame=True, parser="auto")
    if target_col is None:
        X_df = data.data.copy()
        y = pd.to_numeric(data.target, errors="coerce")
        df = X_df.copy()
        df["__target__"] = y
        X, y_arr = frame_to_arrays(df, "__target__")
    else:
        df = data.frame.copy()
        X, y_arr = frame_to_arrays(df, target_col)
    version_text = "" if version is None else f", version={version}"
    return X, y_arr, f"sklearn.datasets.fetch_openml({dataset_name}{version_text})"


def load_real_world_dataset(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a real-world regression dataset with no synthetic fallback."""
    if name == "diabetes":
        data = load_diabetes()
        X = data.data.astype(np.float64)
        y = data.target.astype(np.float64)
        source = "sklearn.datasets.load_diabetes"
    elif name == "california_housing":
        try:
            data = fetch_california_housing()
            X = data.data.astype(np.float64)
            y = data.target.astype(np.float64)
            source = "sklearn.datasets.fetch_california_housing"
        except Exception:
            X, y, source = openml_to_arrays(
                "california_housing",
                target_col="median_house_value",
                version=1,
            )
    elif name == "wine_quality":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=";")
        y = df["quality"].to_numpy(dtype=np.float64)
        X = df.drop(columns=["quality"]).to_numpy(dtype=np.float64)
        source = url
    elif name == "airfoil":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
        df = pd.read_csv(url, sep="\t", header=None)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
        y = df.iloc[:, -1].to_numpy(dtype=np.float64)
        source = url
    elif name == "abalone":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        cols = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight",
            "Rings",
        ]
        df = pd.read_csv(url, header=None, names=cols)
        df["Sex"] = df["Sex"].map({"M": 0, "F": 1, "I": 2}).astype(float)
        y = df["Rings"].to_numpy(dtype=np.float64)
        X = df.drop(columns=["Rings"]).to_numpy(dtype=np.float64)
        source = url
    elif name == "cpu_activity":
        X, y, source = openml_to_arrays("cpu_act")
    elif name == "house_16h":
        X, y, source = openml_to_arrays("house_16H")
    elif name == "insurance":
        url = "https://raw.githubusercontent.com/datagy/data/main/insurance.csv"
        df = pd.read_csv(url)
        X, y = frame_to_arrays(df, "charges")
        source = url
    elif name == "bike_sharing":
        X, y, source = openml_to_arrays("Bike_Sharing_Demand", version=2)
    elif name == "ames_housing":
        X, y, source = openml_to_arrays("ames_housing")
    elif name == "ibm_employee":
        url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
        df = pd.read_csv(url)
        X, y = frame_to_arrays(df, "MonthlyIncome")
        source = url
    elif name == "melbourne_housing":
        url = "https://raw.githubusercontent.com/njtierney/melb-housing-data/master/data/housing.csv"
        df = pd.read_csv(url)
        X, y = frame_to_arrays(df, "price")
        source = url
    else:
        choices = tuple(dict.fromkeys(REAL_WORLD_DATASETS + UAI9_DATASETS))
        raise ValueError(f"Unknown real-world dataset '{name}'. Choose from {choices}.")

    if len(y) < 100:
        raise ValueError(f"Dataset '{name}' has too few rows for the pilot: {len(y)}.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"Dataset '{name}' contains non-finite values.")

    manifest = {
        "dataset": name,
        "source": source,
        "status": "loaded",
        "fallback_used": False,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
    }
    return X, y, manifest


def chronological_split_scale(
    X: np.ndarray, y: np.ndarray, test_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = int(len(y) * (1.0 - test_fraction))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, y_train, X_test, y_test


def iter_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for start in range(0, len(y), batch_size):
        end = min(start + batch_size, len(y))
        yield X[start:end], y[start:end]


def no_rebalance(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    return X, y, None, {"total_real": int(len(y)), "total_replay": 0}


def static_tail_weight(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    tail_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    weights = np.ones(len(y), dtype=np.float64)
    weights[fixed_tail_mask(y, thresholds)] = 1.0 + tail_weight
    return X, y, weights, {
        "total_real": int(len(y)),
        "total_replay": 0,
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
    }


def random_tail_replay(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    replay_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    rng = np.random.default_rng(seed)
    rare_idx = np.where(fixed_tail_mask(y, thresholds))[0]
    if len(rare_idx) == 0 or replay_ratio <= 0:
        return X, y, None, {"total_real": int(len(y)), "total_replay": 0}
    n_replay = int(replay_ratio * len(y))
    choices = rng.choice(rare_idx, size=n_replay, replace=True)
    X_out = np.vstack([X, X[choices]])
    y_out = np.concatenate([y, y[choices]])
    return X_out, y_out, None, {
        "total_real": int(len(y)),
        "total_replay": int(n_replay),
    }


def inverse_density_weight(
    X: np.ndarray,
    y: np.ndarray,
    bins: int = 12,
    max_weight: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Histogram-density weighted ERM baseline."""
    counts, edges = np.histogram(y, bins=bins)
    bin_idx = np.clip(np.digitize(y, edges[1:-1], right=False), 0, len(counts) - 1)
    density = counts[bin_idx].astype(np.float64)
    weights = np.median(counts[counts > 0]) / np.maximum(density, 1.0)
    weights = np.clip(weights, 1.0, max_weight)
    return X, y, weights, {
        "total_real": int(len(y)),
        "total_replay": 0,
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
    }


def smogn_batch(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    target_ratio: float = 0.60,
    noise_scale: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Small SMOGN-style non-streaming batch baseline for pilot comparison."""
    rng = np.random.default_rng(seed)
    rare_idx = np.where(fixed_tail_mask(y, thresholds))[0]
    n_rare = len(rare_idx)
    if n_rare == 0:
        return X, y, None, {"total_real": int(len(y)), "total_replay": 0}

    n_target_rare = int(target_ratio * len(y))
    n_new = max(0, n_target_rare - n_rare)
    if n_new == 0:
        return X, y, None, {"total_real": int(len(y)), "total_replay": 0}

    choices = rng.choice(rare_idx, size=n_new, replace=True)
    x_std = np.std(X, axis=0, keepdims=True)
    y_std = max(float(np.std(y)), 1e-8)
    X_s = X[choices] + rng.normal(0.0, noise_scale, size=(n_new, X.shape[1])) * x_std
    y_s = y[choices] + rng.normal(0.0, noise_scale * y_std, size=n_new)
    X_out = np.vstack([X, X_s])
    y_out = np.concatenate([y, y_s])
    return X_out, y_out, None, {
        "total_real": int(len(y)),
        "total_replay": int(n_new),
    }


def apply_candidate_route(
    route: str,
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    cfg: RouterConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    if route == "none":
        return no_rebalance(X, y)
    if route == "static":
        return static_tail_weight(X, y, thresholds, tail_weight=min(cfg.tail_weight, 2.0))
    if route == "density":
        return inverse_density_weight(X, y, max_weight=min(cfg.max_weight, 8.0))
    if route == "random_replay":
        return random_tail_replay(X, y, thresholds, replay_ratio=min(cfg.replay_ratio, 0.25), seed=seed)
    if route == "smogn":
        return smogn_batch(X, y, thresholds, seed=seed, target_ratio=0.45, noise_scale=0.02)
    if route == "sketch_weight":
        mild_cfg = RouterConfig(**{**asdict(cfg), "tail_weight": 2.0, "deficit_weight": 1.0, "drift_weight": 0.25, "replay_ratio": 0.0, "max_weight": 6.0})
        return sketch_weighted_tail_router(X, y, thresholds, mild_cfg, seed)
    if route == "sketch_replay":
        mild_cfg = RouterConfig(**{**asdict(cfg), "tail_weight": 2.0, "deficit_weight": 1.0, "drift_weight": 0.25, "replay_ratio": 0.12, "max_weight": 6.0})
        return sketch_weighted_tail_router(X, y, thresholds, mild_cfg, seed)
    raise ValueError(route)


def calibrated_sketch_tail_router(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    cfg: RouterConfig,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Select a router action using only a chronological training calibration slice."""
    split = int(0.78 * len(y))
    if split < 100 or len(y) - split < 30:
        X_fit, y_fit, weights, stats = no_rebalance(X, y)
        stats["selected_route"] = "none_too_small"
        return X_fit, y_fit, weights, stats

    X_base, y_base = X[:split], y[:split]
    X_cal, y_cal = X[split:], y[split:]
    routes = ("none", "static", "density", "random_replay", "smogn", "sketch_weight", "sketch_replay")
    scores: dict[str, dict[str, float]] = {}

    best_route = "none"
    best_score = float("inf")
    base_mse = None
    base_rare = None
    for route in routes:
        X_c, y_c, w_c, _ = apply_candidate_route(route, X_base, y_base, thresholds, cfg, seed)
        metrics = evaluate_downstream(
            y_c,
            X_c,
            y_cal,
            X_cal,
            model_type="xgboost",
            n_estimators=max(40, min(n_estimators, 90)),
            seed=seed,
            sample_weight=w_c,
            rare_thresholds=thresholds,
        )
        scores[route] = {"mse": float(metrics["mse"]), "mse_rare": float(metrics["mse_rare"])}
        if route == "none":
            base_mse = float(metrics["mse"])
            base_rare = float(metrics["mse_rare"])

    assert base_mse is not None and base_rare is not None
    min_calibration_rare_gain = 0.05
    max_calibration_overall_loss = 0.08
    for route, score in scores.items():
        if route == "none":
            continue
        rare_gain = (base_rare - score["mse_rare"]) / max(base_rare, 1e-12)
        overall_loss = (score["mse"] - base_mse) / max(base_mse, 1e-12)
        if rare_gain < min_calibration_rare_gain:
            continue
        if overall_loss > max_calibration_overall_loss:
            continue
        objective = score["mse_rare"] + 0.10 * score["mse"]
        if objective < best_score:
            best_score = objective
            best_route = route

    X_fit, y_fit, weights, stats = apply_candidate_route(best_route, X, y, thresholds, cfg, seed)
    stats["selected_route"] = best_route
    stats["calibration_scores"] = scores
    return X_fit, y_fit, weights, stats


def sketch_weighted_tail_router(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    cfg: RouterConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    ref_tail_mass = cfg.low_q + (1.0 - cfg.high_q)
    ref_std = max(float(np.std(y)), 1e-8)
    ema_mean = float(np.mean(y[: min(len(y), cfg.window_size)]))
    recent_tail: list[bool] = []
    replay_X: list[np.ndarray] = []
    replay_y: list[float] = []
    out_X: list[np.ndarray] = []
    out_y: list[np.ndarray] = []
    out_w: list[np.ndarray] = []
    stats = RouterStats()

    for X_b, y_b in iter_batches(X, y, cfg.batch_size):
        rare_b = fixed_tail_mask(y_b, thresholds)
        if recent_tail:
            current_tail_mass = float(np.mean(recent_tail[-cfg.window_size:]))
        else:
            current_tail_mass = float(rare_b.mean()) if len(rare_b) else 0.0
        deficit = max(0.0, (ref_tail_mass - current_tail_mass) / max(ref_tail_mass, 1e-8))

        batch_mean = float(np.mean(y_b))
        drift_score = min(2.0, abs(batch_mean - ema_mean) / ref_std)
        ema_mean = 0.95 * ema_mean + 0.05 * batch_mean

        weights = np.ones(len(y_b), dtype=np.float64)
        rare_multiplier = 1.0 + cfg.tail_weight + cfg.deficit_weight * deficit + cfg.drift_weight * drift_score
        weights[rare_b] = np.minimum(cfg.max_weight, rare_multiplier)

        out_X.append(X_b)
        out_y.append(y_b)
        out_w.append(weights)

        if replay_X and deficit > 0 and cfg.replay_ratio > 0:
            n_replay = min(
                len(replay_y),
                int(np.ceil(cfg.replay_ratio * len(y_b) * deficit)),
            )
            if n_replay > 0:
                choices = rng.choice(len(replay_y), size=n_replay, replace=True)
                X_r = np.vstack([replay_X[i] for i in choices])
                y_r = np.array([replay_y[i] for i in choices], dtype=np.float64)
                w_r = np.full(n_replay, min(cfg.max_weight, rare_multiplier), dtype=np.float64)
                out_X.append(X_r)
                out_y.append(y_r)
                out_w.append(w_r)
                stats.total_replay += int(n_replay)

        for row, target, is_rare in zip(X_b, y_b, rare_b):
            recent_tail.append(bool(is_rare))
            if is_rare:
                replay_X.append(row.copy())
                replay_y.append(float(target))

        if len(recent_tail) > cfg.window_size:
            recent_tail = recent_tail[-cfg.window_size:]
        if len(replay_y) > cfg.window_size:
            replay_X = replay_X[-cfg.window_size:]
            replay_y = replay_y[-cfg.window_size:]

        stats.total_real += int(len(y_b))
        stats.mean_tail_mass += current_tail_mass
        stats.mean_deficit += deficit
        stats.mean_weight += float(weights.mean())
        stats.max_weight = max(stats.max_weight, float(weights.max()))
        stats.n_batches += 1

    X_out = np.vstack(out_X)
    y_out = np.concatenate(out_y)
    w_out = np.concatenate(out_w)
    denom = max(stats.n_batches, 1)
    summary = asdict(stats)
    summary["mean_tail_mass"] = stats.mean_tail_mass / denom
    summary["mean_deficit"] = stats.mean_deficit / denom
    summary["mean_weight"] = stats.mean_weight / denom
    return X_out, y_out, w_out, summary


def evaluate_method(
    dataset: str,
    method: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: tuple[float, float],
    cfg: RouterConfig,
    n_estimators: int,
) -> dict:
    t0 = time.perf_counter()
    if method == "No-Rebalance":
        X_fit, y_fit, weights, stats = no_rebalance(X_train, y_train)
    elif method == "No-Routing-Parity":
        X_fit, y_fit, weights, stats = no_rebalance(X_train, y_train)
    elif method == "StaticTailWeight":
        X_fit, y_fit, weights, stats = static_tail_weight(X_train, y_train, thresholds, cfg.tail_weight)
    elif method == "InverseDensityWeight":
        X_fit, y_fit, weights, stats = inverse_density_weight(
            X_train, y_train, max_weight=min(cfg.max_weight, 10.0)
        )
    elif method == "RandomTailReplay":
        X_fit, y_fit, weights, stats = random_tail_replay(
            X_train, y_train, thresholds, cfg.replay_ratio, seed
        )
    elif method == "SMOGN-Batch":
        X_fit, y_fit, weights, stats = smogn_batch(X_train, y_train, thresholds, seed)
    elif method == "SketchWeightOnly":
        weight_only_cfg = RouterConfig(**{**asdict(cfg), "replay_ratio": 0.0})
        X_fit, y_fit, weights, stats = sketch_weighted_tail_router(
            X_train, y_train, thresholds, weight_only_cfg, seed
        )
    elif method == "SketchTailRouter":
        X_fit, y_fit, weights, stats = calibrated_sketch_tail_router(
            X_train, y_train, thresholds, cfg, seed, n_estimators
        )
    else:
        raise ValueError(method)

    metrics = evaluate_downstream(
        y_fit,
        X_fit,
        y_test,
        X_test,
        model_type="xgboost",
        n_estimators=n_estimators,
        seed=seed,
        sample_weight=weights,
        rare_thresholds=thresholds,
    )
    elapsed = time.perf_counter() - t0
    ir_before = compute_imbalance_ratio_fixed(y_train, thresholds)
    ir_after = compute_imbalance_ratio_fixed(y_fit, thresholds)
    row = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "mse_rare": metrics["mse_rare"],
        "mae_rare": metrics["mae_rare"],
        "f1_rare": metrics["f1_rare"],
        "ir_before": ir_before,
        "ir_after": ir_after,
        "n_train": len(y_train),
        "n_fit": len(y_fit),
        "n_replay": int(stats.get("total_replay", 0)),
        "mean_weight": float(stats.get("mean_weight", 1.0)),
        "max_weight": float(stats.get("max_weight", 1.0)),
        "selected_route": str(stats.get("selected_route", "")),
        "elapsed_s": elapsed,
    }
    return row


def summarize(rows: list[dict], parity_pass: bool, output_dir: Path) -> dict:
    datasets = sorted({r["dataset"] for r in rows})
    summary_rows = []
    router_wins = 0
    router_ties_or_wins = 0
    catastrophic = 0
    for dataset in datasets:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        by_method: dict[str, dict[str, float]] = {}
        for method in sorted({r["method"] for r in ds_rows}):
            m_rows = [r for r in ds_rows if r["method"] == method]
            by_method[method] = {
                "mse": float(np.mean([r["mse"] for r in m_rows])),
                "mse_rare": float(np.mean([r["mse_rare"] for r in m_rows])),
                "mae_rare": float(np.mean([r["mae_rare"] for r in m_rows])),
                "f1_rare": float(np.mean([r["f1_rare"] for r in m_rows])),
            }
        base = by_method["No-Rebalance"]
        router = by_method["SketchTailRouter"]
        rare_delta = (router["mse_rare"] - base["mse_rare"]) / max(base["mse_rare"], 1e-12)
        mse_delta = (router["mse"] - base["mse"]) / max(base["mse"], 1e-12)
        best_rare = min(v["mse_rare"] for v in by_method.values())
        if router["mse_rare"] <= best_rare * 1.001:
            router_wins += 1
        if rare_delta <= 0.02:
            router_ties_or_wins += 1
        if mse_delta > 0.25:
            catastrophic += 1
        summary_rows.append({
            "dataset": dataset,
            "router_mse_rare": router["mse_rare"],
            "baseline_mse_rare": base["mse_rare"],
            "rare_delta_pct": 100.0 * rare_delta,
            "overall_mse_delta_pct": 100.0 * mse_delta,
            "router_best_rare": router["mse_rare"] <= best_rare * 1.001,
        })

    pass_gates = {
        "parity_pass": parity_pass,
        "rare_tie_or_win_majority": router_ties_or_wins >= max(1, int(np.ceil(0.67 * len(datasets)))),
        "no_catastrophic_overall_loss": catastrophic == 0,
    }
    payload = {
        "pass": bool(all(pass_gates.values())),
        "pass_gates": pass_gates,
        "router_best_rare_datasets": router_wins,
        "router_tie_or_win_datasets": router_ties_or_wins,
        "n_datasets": len(datasets),
        "dataset_summary": summary_rows,
    }
    (output_dir / "pilot_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sketch-weighted router pilot.")
    parser.add_argument("--mode", choices=["synthetic", "real", "uai9", "both"], default="real")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--output-dir", default="results/pilot_router_real")
    parser.add_argument("--n", type=int, default=1800)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--n-estimators", type=int, default=180)
    parser.add_argument("--tail-weight", type=float, default=5.0)
    parser.add_argument("--deficit-weight", type=float, default=4.0)
    parser.add_argument("--drift-weight", type=float, default=1.5)
    parser.add_argument("--replay-ratio", type=float, default=0.35)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RouterConfig(
        tail_weight=args.tail_weight,
        deficit_weight=args.deficit_weight,
        drift_weight=args.drift_weight,
        replay_ratio=args.replay_ratio,
    )
    methods = [
        "No-Rebalance",
        "No-Routing-Parity",
        "StaticTailWeight",
        "InverseDensityWeight",
        "RandomTailReplay",
        "SMOGN-Batch",
        "SketchWeightOnly",
        "SketchTailRouter",
    ]
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    parity_pass = True

    if args.datasets is not None:
        datasets = args.datasets
    elif args.mode == "synthetic":
        datasets = list(PILOT_DATASETS)
    elif args.mode == "real":
        datasets = list(REAL_WORLD_DATASETS)
    elif args.mode == "uai9":
        datasets = list(UAI9_DATASETS)
    else:
        datasets = list(PILOT_DATASETS) + list(REAL_WORLD_DATASETS)

    print("=" * 72)
    print("TabStream-IR sketch-weighted tail-router pilot")
    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Seeds: {args.seeds}")
    print(f"Router config: {cfg}")

    real_dataset_names = tuple(dict.fromkeys(REAL_WORLD_DATASETS + UAI9_DATASETS))
    for dataset in datasets:
        is_real = dataset in real_dataset_names
        if is_real:
            try:
                X_real, y_real, manifest = load_real_world_dataset(dataset)
            except Exception as exc:
                manifest_rows.append({
                    "dataset": dataset,
                    "status": "failed",
                    "fallback_used": False,
                    "error": str(exc),
                })
                print(f"\n[{dataset}] SKIP real dataset load failed: {exc}")
                continue
            manifest_rows.append(manifest)
        for seed in args.seeds:
            if is_real:
                X, y = X_real, y_real
            else:
                X, y = make_pilot_dataset(dataset, seed=seed, n=args.n)
            X_train, y_train, X_test, y_test = chronological_split_scale(X, y, args.test_fraction)
            thresholds = tail_thresholds(y_train, cfg.low_q, cfg.high_q)

            raw_X, raw_y, raw_w, _ = no_rebalance(X_train, y_train)
            parity_X, parity_y, parity_w, _ = no_rebalance(X_train, y_train)
            this_parity = (
                np.array_equal(raw_X, parity_X)
                and np.array_equal(raw_y, parity_y)
                and raw_w is None
                and parity_w is None
            )
            parity_pass = parity_pass and this_parity

            print(f"\n[{dataset} seed={seed}] fixed_tail={thresholds} parity={this_parity}")
            for method in methods:
                row = evaluate_method(
                    dataset,
                    method,
                    seed,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    thresholds,
                    cfg,
                    args.n_estimators,
                )
                rows.append(row)
                print(
                    f"  {method:<18} rare_mse={row['mse_rare']:.4f} "
                    f"rare_mae={row['mae_rare']:.4f} mse={row['mse']:.4f} "
                    f"IR={row['ir_before']:.2f}->{row['ir_after']:.2f} "
                    f"n_fit={row['n_fit']} w_mean={row['mean_weight']:.2f}"
                )

    csv_path = output_dir / "pilot_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    summary = summarize(rows, parity_pass, output_dir)
    print("\n" + "=" * 72)
    print(f"Saved: {csv_path}")
    print(f"Saved: {manifest_path}")
    print(f"Saved: {output_dir / 'pilot_summary.json'}")
    print(f"PASS: {summary['pass']}  gates={summary['pass_gates']}")
    print("=" * 72)

    if not summary["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
