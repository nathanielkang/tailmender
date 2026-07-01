#!/usr/bin/env python3
"""Evaluate TailMender against common imbalanced-regression baselines on tabular benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from run_router_pilot import (  # noqa: E402
    UAI9_DATASETS,
    chronological_split_scale,
    fixed_tail_mask,
    inverse_density_weight,
    load_real_world_dataset,
    random_tail_replay,
    smogn_batch,
    static_tail_weight,
)
from src.evaluation.metrics import (  # noqa: E402
    mae_rare_fixed,
    mse_rare_fixed,
    tail_thresholds,
)
from deep_baselines import (  # noqa: E402
    fit_predict_conr,
    fit_predict_ranksim,
    fit_predict_vir,
    torch_available,
)


PRIMARY_METHODS = ("TailMender",)


def make_model(seed: int, n_estimators: int):
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            verbosity=0,
            n_jobs=1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=0.06,
            random_state=seed,
        )


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def cap_chronological(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_train: int,
    max_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y_train) > max_train:
        X_train = X_train[-max_train:]
        y_train = y_train[-max_train:]
    if len(y_test) > max_test:
        X_test = X_test[:max_test]
        y_test = y_test[:max_test]
    return X_train, y_train, X_test, y_test


def resolve_caps(profile: str, max_train: int | None, max_test: int | None) -> tuple[int | None, int | None]:
    if profile == "smoke":
        return 2500 if max_train is None else max_train, 1200 if max_test is None else max_test
    return max_train, max_test


def _masked_mse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def smoter_batch(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    target_ratio: float = 0.60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """SMOTER-style interpolation baseline for rare regression targets."""
    rng = np.random.default_rng(seed)
    rare_idx = np.where(fixed_tail_mask(y, thresholds))[0]
    n_rare = len(rare_idx)
    if n_rare < 2:
        return X, y, None, {"total_real": int(len(y)), "total_replay": 0}
    n_target_rare = int(target_ratio * len(y))
    n_new = max(0, n_target_rare - n_rare)
    if n_new == 0:
        return X, y, None, {"total_real": int(len(y)), "total_replay": 0}
    left = rng.choice(rare_idx, size=n_new, replace=True)
    right = rng.choice(rare_idx, size=n_new, replace=True)
    alpha = rng.uniform(0.0, 1.0, size=(n_new, 1))
    X_s = X[left] + alpha * (X[right] - X[left])
    y_s = y[left] + alpha.ravel() * (y[right] - y[left])
    return np.vstack([X, X_s]), np.concatenate([y, y_s]), None, {
        "total_real": int(len(y)),
        "total_replay": int(n_new),
    }


def lds_weight(
    X: np.ndarray,
    y: np.ndarray,
    bins: int = 24,
    max_weight: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Label-distribution smoothing weight baseline following DIR practice."""
    counts, edges = np.histogram(y, bins=bins)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
    bin_idx = np.clip(np.digitize(y, edges[1:-1], right=False), 0, len(counts) - 1)
    density = np.maximum(smooth[bin_idx], 1.0)
    weights = np.sqrt(np.median(smooth[smooth > 0]) / density)
    weights = np.clip(weights, 1.0, max_weight)
    return X, y, weights, {
        "total_real": int(len(y)),
        "total_replay": 0,
        "mean_weight": float(weights.mean()),
        "max_weight": float(weights.max()),
    }


def predict_rebagg_resample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, dict]:
    """REBAGG-style ensemble with varying rare/normal resampled training sets."""
    rng = np.random.default_rng(seed)
    rare_idx = np.where(fixed_tail_mask(y_train, thresholds))[0]
    bulk_idx = np.where(~fixed_tail_mask(y_train, thresholds))[0]
    if len(rare_idx) == 0 or len(bulk_idx) == 0:
        model = make_model(seed + 4101, n_estimators)
        model.fit(X_train, y_train)
        return model.predict(X_test), {"n_fit": len(y_train), "ensemble_size": 1}
    ratios = (1 / 3, 0.4, 0.5, 0.6, 2 / 3)
    preds = []
    for i, ratio in enumerate(ratios):
        n_total = len(y_train)
        n_rare = max(1, int(ratio * n_total))
        n_bulk = max(1, n_total - n_rare)
        sample_idx = np.concatenate(
            [
                rng.choice(rare_idx, size=n_rare, replace=True),
                rng.choice(bulk_idx, size=n_bulk, replace=True),
            ]
        )
        rng.shuffle(sample_idx)
        model = make_model(seed + 4201 + i, max(10, n_estimators // 2))
        model.fit(X_train[sample_idx], y_train[sample_idx])
        preds.append(model.predict(X_test))
    return np.mean(np.vstack(preds), axis=0), {
        "n_fit": int(len(y_train)),
        "ensemble_size": int(len(ratios)),
    }


def build_tail_atlas(
    X: np.ndarray,
    y: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    max_modes: int = 3,
) -> dict:
    rare_mask = fixed_tail_mask(y, thresholds)
    if rare_mask.sum() < max_modes:
        centers = X[rare_mask].copy() if rare_mask.any() else X[:1].copy()
    else:
        k = min(max_modes, int(rare_mask.sum()))
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        km.fit(X[rare_mask])
        centers = km.cluster_centers_
    return {
        "centers": centers,
        "x_scale": np.maximum(np.std(X, axis=0), 1e-6),
    }


def tail_atlas_features(X: np.ndarray, atlas: dict) -> tuple[np.ndarray, np.ndarray]:
    centers = atlas["centers"]
    scale = atlas["x_scale"]
    distances = []
    for center in centers:
        d = np.sqrt(np.mean(((X - center) / scale) ** 2, axis=1))
        distances.append(d)
    D = np.vstack(distances).T if distances else np.zeros((len(X), 1))
    min_dist = D.min(axis=1)
    score = np.exp(-min_dist)
    features = np.column_stack([X, score, min_dist, D.mean(axis=1)])
    return features, score


def _residual_zero_targets_and_weights(
    y: np.ndarray,
    pred_teacher: np.ndarray,
    thresholds: tuple[float, float],
    rare_pressure: float,
    bulk_zero_weight: float,
    near_tail_margin: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    lo, hi = thresholds
    span = max(float(hi - lo), 1e-8)
    rare_mask = fixed_tail_mask(y, thresholds)
    near_tail = (y <= lo + near_tail_margin * span) | (y >= hi - near_tail_margin * span)
    train_residual_mask = rare_mask | near_tail
    bulk_safe_mask = ~train_residual_mask
    targets = np.zeros(len(y), dtype=np.float64)
    targets[train_residual_mask] = y[train_residual_mask] - pred_teacher[train_residual_mask]
    weights = np.ones(len(y), dtype=np.float64)
    weights[bulk_safe_mask] = bulk_zero_weight
    weights[train_residual_mask] = max(1.0, rare_pressure * 0.50)
    weights[rare_mask] = rare_pressure
    return targets, weights, {
        "n_rare": int(rare_mask.sum()),
        "n_near_tail": int(train_residual_mask.sum()),
        "n_bulk_safe": int(bulk_safe_mask.sum()),
    }


def fit_tailmender_single(
    X_train: np.ndarray,
    y_train: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    n_estimators: int,
) -> dict:
    split = int(0.76 * len(y_train))
    if split < 120 or len(y_train) - split < 40:
        split = int(0.70 * len(y_train))
    X_base, y_base = X_train[:split], y_train[:split]
    X_cal, y_cal = X_train[split:], y_train[split:]
    rare_cal = fixed_tail_mask(y_cal, thresholds)
    bulk_cal = ~rare_cal

    teacher_cal = make_model(seed + 1801, n_estimators)
    teacher_cal.fit(X_base, y_base)
    pred_teacher_base = teacher_cal.predict(X_base)
    pred_teacher_cal = teacher_cal.predict(X_cal)
    base_rare_mse = mse_rare_fixed(y_cal, pred_teacher_cal, thresholds)
    base_bulk_mse = _masked_mse(y_cal, pred_teacher_cal, bulk_cal)
    base_bulk_mae = _masked_mae(y_cal, pred_teacher_cal, bulk_cal)
    base_mse = float(np.mean((y_cal - pred_teacher_cal) ** 2))
    best = {
        "use_atlas": False,
        "rare_pressure": 0.0,
        "bulk_zero_weight": 0.0,
        "near_tail_margin": 0.0,
        "objective": float("inf"),
        "source": "teacher_only",
    }

    for use_atlas in (False, True):
        if use_atlas:
            atlas_cal = build_tail_atlas(X_base, y_base, thresholds, seed + 1811, max_modes=3)
            X_student_base, _ = tail_atlas_features(X_base, atlas_cal)
            X_student_cal, _ = tail_atlas_features(X_cal, atlas_cal)
        else:
            X_student_base, X_student_cal = X_base, X_cal
        idx = 0
        for rare_pressure in (3.0, 6.0, 10.0, 14.0):
            for bulk_zero_weight in (2.0, 6.0, 12.0):
                for near_tail_margin in (0.08, 0.16, 0.26):
                    idx += 1
                    targets, weights, _ = _residual_zero_targets_and_weights(
                        y_base, pred_teacher_base, thresholds, rare_pressure, bulk_zero_weight, near_tail_margin
                    )
                    student = make_model(seed + 1821 + idx + (300 if use_atlas else 0), max(40, n_estimators))
                    student.fit(X_student_base, targets, sample_weight=weights)
                    pred = pred_teacher_cal + student.predict(X_student_cal)
                    rare_mse = mse_rare_fixed(y_cal, pred, thresholds)
                    bulk_mse = _masked_mse(y_cal, pred, bulk_cal)
                    bulk_mae = _masked_mae(y_cal, pred, bulk_cal)
                    mse = float(np.mean((y_cal - pred) ** 2))
                    rare_gain = (base_rare_mse - rare_mse) / max(base_rare_mse, 1e-8)
                    bulk_mae_loss = (bulk_mae - base_bulk_mae) / max(base_bulk_mae, 1e-8)
                    bulk_mse_loss = (bulk_mse - base_bulk_mse) / max(base_bulk_mse, 1e-8)
                    if rare_cal.sum() >= 8 and rare_gain < 0.01:
                        continue
                    if bulk_mae_loss > 0.30 or bulk_mse_loss > 0.45:
                        continue
                    objective = rare_mse / max(base_rare_mse, 1e-8) + 0.30 * bulk_mae / max(
                        base_bulk_mae, 1e-8
                    ) + 0.12 * mse / max(base_mse, 1e-8)
                    if objective < best["objective"]:
                        best = {
                            "use_atlas": bool(use_atlas),
                            "rare_pressure": float(rare_pressure),
                            "bulk_zero_weight": float(bulk_zero_weight),
                            "near_tail_margin": float(near_tail_margin),
                            "objective": float(objective),
                            "source": "single_residual_student",
                        }

    teacher = make_model(seed + 1901, n_estimators)
    teacher.fit(X_train, y_train)
    if best["source"] == "teacher_only":
        return {
            "teacher": teacher,
            "students": [],
            "configs": [],
            "atlas_by_use": {},
            "aggregation": "none",
            "mode": "teacher_only",
            "n_fit": len(y_train),
        }
    pred_teacher_train = teacher.predict(X_train)
    atlas_by_use: dict[bool, object] = {}
    if best["use_atlas"]:
        atlas = build_tail_atlas(X_train, y_train, thresholds, seed + 1911, max_modes=3)
        X_student_train, _ = tail_atlas_features(X_train, atlas)
        atlas_by_use[True] = atlas
    else:
        X_student_train = X_train
    targets, weights, _ = _residual_zero_targets_and_weights(
        y_train,
        pred_teacher_train,
        thresholds,
        float(best["rare_pressure"]),
        float(best["bulk_zero_weight"]),
        float(best["near_tail_margin"]),
    )
    student = make_model(seed + 1921, max(40, n_estimators))
    student.fit(X_student_train, targets, sample_weight=weights)
    return {
        "teacher": teacher,
        "students": [student],
        "configs": [
            {
                "use_atlas": bool(best["use_atlas"]),
                "rare_pressure": float(best["rare_pressure"]),
                "bulk_zero_weight": float(best["bulk_zero_weight"]),
                "near_tail_margin": float(best["near_tail_margin"]),
            }
        ],
        "atlas_by_use": atlas_by_use,
        "aggregation": "single",
        "mode": "single",
        "n_fit": len(y_train),
    }


def _tailmender_candidate_configs() -> list[dict]:
    return [
        {"use_atlas": False, "rare_pressure": 4.0, "bulk_zero_weight": 4.0, "near_tail_margin": 0.10},
        {"use_atlas": False, "rare_pressure": 7.0, "bulk_zero_weight": 6.0, "near_tail_margin": 0.16},
        {"use_atlas": False, "rare_pressure": 11.0, "bulk_zero_weight": 10.0, "near_tail_margin": 0.22},
        {"use_atlas": False, "rare_pressure": 15.0, "bulk_zero_weight": 14.0, "near_tail_margin": 0.30},
        {"use_atlas": True, "rare_pressure": 5.0, "bulk_zero_weight": 4.0, "near_tail_margin": 0.10},
        {"use_atlas": True, "rare_pressure": 8.0, "bulk_zero_weight": 7.0, "near_tail_margin": 0.18},
        {"use_atlas": True, "rare_pressure": 12.0, "bulk_zero_weight": 10.0, "near_tail_margin": 0.24},
        {"use_atlas": True, "rare_pressure": 16.0, "bulk_zero_weight": 14.0, "near_tail_margin": 0.30},
    ]


def fit_tailmender(
    X_train: np.ndarray,
    y_train: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    n_estimators: int,
) -> dict:
    """Bulk teacher plus residual-zero student ensemble with small-window fallback."""
    if len(y_train) < 800:
        return fit_tailmender_single(X_train, y_train, thresholds, seed, n_estimators)

    split = int(0.76 * len(y_train))
    X_base, y_base = X_train[:split], y_train[:split]
    X_cal, y_cal = X_train[split:], y_train[split:]
    bulk_cal = ~fixed_tail_mask(y_cal, thresholds)
    teacher_cal = make_model(seed + 2001, n_estimators)
    teacher_cal.fit(X_base, y_base)
    pred_teacher_base = teacher_cal.predict(X_base)
    pred_teacher_cal = teacher_cal.predict(X_cal)
    base_rare_mse = mse_rare_fixed(y_cal, pred_teacher_cal, thresholds)
    base_bulk_mae = _masked_mae(y_cal, pred_teacher_cal, bulk_cal)
    base_mse = float(np.mean((y_cal - pred_teacher_cal) ** 2))

    atlas_cache: dict[bool, tuple[object | None, np.ndarray, np.ndarray]] = {False: (None, X_base, X_cal)}
    candidates: list[dict] = []
    best = {"objective": float("inf"), "indices": [], "aggregation": "none", "mode": "teacher_only"}
    for idx, config in enumerate(_tailmender_candidate_configs()):
        use_atlas = bool(config["use_atlas"])
        if use_atlas not in atlas_cache:
            atlas = build_tail_atlas(X_base, y_base, thresholds, seed + 2011, max_modes=3)
            X_student_base, _ = tail_atlas_features(X_base, atlas)
            X_student_cal, _ = tail_atlas_features(X_cal, atlas)
            atlas_cache[use_atlas] = (atlas, X_student_base, X_student_cal)
        _, X_student_base, X_student_cal = atlas_cache[use_atlas]
        targets, weights, _ = _residual_zero_targets_and_weights(
            y_base,
            pred_teacher_base,
            thresholds,
            float(config["rare_pressure"]),
            float(config["bulk_zero_weight"]),
            float(config["near_tail_margin"]),
        )
        student = make_model(seed + 2021 + idx * 17, max(45, n_estimators))
        student.fit(X_student_base, targets, sample_weight=weights)
        residual_cal = student.predict(X_student_cal)
        pred = pred_teacher_cal + residual_cal
        rare_mse = mse_rare_fixed(y_cal, pred, thresholds)
        bulk_mae = _masked_mae(y_cal, pred, bulk_cal)
        mse = float(np.mean((y_cal - pred) ** 2))
        objective = rare_mse / max(base_rare_mse, 1e-8) + 0.32 * bulk_mae / max(
            base_bulk_mae, 1e-8
        ) + 0.12 * mse / max(base_mse, 1e-8)
        candidate = {"index": idx, "config": config, "residual_cal": residual_cal, "objective": float(objective)}
        candidates.append(candidate)
        if objective < best["objective"]:
            best = {"objective": float(objective), "indices": [idx], "aggregation": "single", "mode": "single"}

    ranked = sorted(candidates, key=lambda c: c["objective"])
    for size in (3, 5):
        pool = ranked[: min(size, len(ranked))]
        if len(pool) < 2:
            continue
        stack = np.vstack([c["residual_cal"] for c in pool])
        for aggregation, residual_cal in (("mean", stack.mean(axis=0)), ("median", np.median(stack, axis=0))):
            pred = pred_teacher_cal + residual_cal
            rare_mse = mse_rare_fixed(y_cal, pred, thresholds)
            bulk_mae = _masked_mae(y_cal, pred, bulk_cal)
            mse = float(np.mean((y_cal - pred) ** 2))
            objective = rare_mse / max(base_rare_mse, 1e-8) + 0.32 * bulk_mae / max(
                base_bulk_mae, 1e-8
            ) + 0.12 * mse / max(base_mse, 1e-8)
            if objective < best["objective"]:
                best = {
                    "objective": float(objective),
                    "indices": [int(c["index"]) for c in pool],
                    "aggregation": aggregation,
                    "mode": "ensemble",
                }

    teacher = make_model(seed + 2101, n_estimators)
    teacher.fit(X_train, y_train)
    if best["mode"] == "teacher_only":
        return {
            "teacher": teacher,
            "students": [],
            "configs": [],
            "atlas_by_use": {},
            "aggregation": "none",
            "mode": "teacher_only",
            "n_fit": len(y_train),
        }
    pred_teacher_train = teacher.predict(X_train)
    selected_configs = [dict(_tailmender_candidate_configs()[idx]) for idx in best["indices"]]
    atlas_by_use: dict[bool, object] = {}
    X_full_cache: dict[bool, np.ndarray] = {False: X_train}
    students = []
    for pos, config in enumerate(selected_configs):
        use_atlas = bool(config["use_atlas"])
        if use_atlas not in X_full_cache:
            atlas = build_tail_atlas(X_train, y_train, thresholds, seed + 2111 + pos, max_modes=3)
            X_student_train, _ = tail_atlas_features(X_train, atlas)
            atlas_by_use[use_atlas] = atlas
            X_full_cache[use_atlas] = X_student_train
        targets, weights, _ = _residual_zero_targets_and_weights(
            y_train,
            pred_teacher_train,
            thresholds,
            float(config["rare_pressure"]),
            float(config["bulk_zero_weight"]),
            float(config["near_tail_margin"]),
        )
        student = make_model(seed + 2121 + pos * 19, max(45, n_estimators))
        student.fit(X_full_cache[use_atlas], targets, sample_weight=weights)
        students.append(student)

    return {
        "teacher": teacher,
        "students": students,
        "configs": selected_configs,
        "atlas_by_use": atlas_by_use,
        "aggregation": best["aggregation"],
        "mode": best["mode"],
        "n_fit": len(y_train),
    }


def predict_tailmender(bundle: dict, X_test: np.ndarray) -> np.ndarray:
    pred_teacher = bundle["teacher"].predict(X_test)
    if not bundle["students"]:
        return pred_teacher
    residuals = []
    X_cache: dict[bool, np.ndarray] = {False: X_test}
    for student, config in zip(bundle["students"], bundle["configs"]):
        use_atlas = bool(config["use_atlas"])
        if use_atlas not in X_cache:
            X_eval, _ = tail_atlas_features(X_test, bundle["atlas_by_use"][use_atlas])
            X_cache[use_atlas] = X_eval
        residuals.append(student.predict(X_cache[use_atlas]))
    stack = np.vstack(residuals)
    if bundle["aggregation"] == "median":
        residual = np.median(stack, axis=0)
    else:
        residual = stack.mean(axis=0)
    return pred_teacher + residual


def fit_predict_method(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    thresholds: tuple[float, float],
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, dict]:
    if method == "No-Rebalance":
        model = make_model(seed, n_estimators)
        model.fit(X_train, y_train)
        return model.predict(X_test), {"n_fit": len(y_train)}
    if method == "StaticTailWeight":
        _, _, weights, stats = static_tail_weight(X_train, y_train, thresholds, tail_weight=3.0)
        model = make_model(seed, n_estimators)
        model.fit(X_train, y_train, sample_weight=weights)
        return model.predict(X_test), {"n_fit": len(y_train), **stats}
    if method == "InverseDensityWeight":
        _, _, weights, stats = inverse_density_weight(X_train, y_train, max_weight=10.0)
        model = make_model(seed, n_estimators)
        model.fit(X_train, y_train, sample_weight=weights)
        return model.predict(X_test), {"n_fit": len(y_train), **stats}
    if method == "LDSWeight":
        _, _, weights, stats = lds_weight(X_train, y_train)
        model = make_model(seed, n_estimators)
        model.fit(X_train, y_train, sample_weight=weights)
        return model.predict(X_test), {"n_fit": len(y_train), **stats}
    if method == "RandomTailReplay":
        X_fit, y_fit, weights, stats = random_tail_replay(X_train, y_train, thresholds, 0.35, seed)
        model = make_model(seed, n_estimators)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model.predict(X_test), {"n_fit": len(y_fit), **stats}
    if method == "SMOTER-Batch":
        X_fit, y_fit, weights, stats = smoter_batch(X_train, y_train, thresholds, seed)
        model = make_model(seed, n_estimators)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model.predict(X_test), {"n_fit": len(y_fit), **stats}
    if method == "SMOGN-Batch":
        X_fit, y_fit, weights, stats = smogn_batch(X_train, y_train, thresholds, seed)
        model = make_model(seed, n_estimators)
        model.fit(X_fit, y_fit, sample_weight=weights)
        return model.predict(X_test), {"n_fit": len(y_fit), **stats}
    if method == "REBAGG-Resample":
        return predict_rebagg_resample(X_train, y_train, X_test, thresholds, seed, n_estimators)
    if method == "RankSim":
        return fit_predict_ranksim(X_train, y_train, X_test, thresholds, seed)
    if method == "ConR":
        return fit_predict_conr(X_train, y_train, X_test, thresholds, seed)
    if method == "VIR":
        return fit_predict_vir(X_train, y_train, X_test, thresholds, seed)
    if method == "TailMender":
        bundle = fit_tailmender(X_train, y_train, thresholds, seed, n_estimators)
        selected_config = {
            "kind": "tailmender",
            "mode": bundle["mode"],
            "aggregation": bundle["aggregation"],
            "ensemble_size": len(bundle["students"]),
            "configs": bundle["configs"],
        }
        return predict_tailmender(bundle, X_test), {
            "n_fit": int(bundle["n_fit"]),
            "selected_config": json.dumps(selected_config),
        }
    raise ValueError(method)


def evaluate_prediction(y_true: np.ndarray, pred: np.ndarray, thresholds: tuple[float, float]) -> dict:
    rare_mask = fixed_tail_mask(y_true, thresholds)
    bulk_mask = ~rare_mask
    return {
        "mse": float(np.mean((y_true - pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - pred))),
        "mse_rare": mse_rare_fixed(y_true, pred, thresholds),
        "mae_rare": mae_rare_fixed(y_true, pred, thresholds),
        "mse_bulk": _masked_mse(y_true, pred, bulk_mask),
        "mae_bulk": _masked_mae(y_true, pred, bulk_mask),
        "rare_count": int(rare_mask.sum()),
        "bulk_count": int(bulk_mask.sum()),
    }


def summarize_rank(rows: list[dict], output_dir: Path, profile: str = "smoke") -> dict:
    import pandas as pd

    df = pd.DataFrame(rows)
    rank_payload = {}
    dataset_ranks = {}
    for metric in ("mse_rare", "mae_rare", "mse_bulk", "mae_bulk", "mse", "mae"):
        means = df.groupby(["dataset", "method"], as_index=False)[metric].mean()
        means["rank"] = means.groupby("dataset")[metric].rank(method="average", ascending=True)
        dataset_ranks[metric] = means.sort_values(["dataset", "rank"]).to_dict("records")
        rank_payload[metric] = (
            means.groupby("method")["rank"].agg(["mean", "median"]).sort_values("mean").to_dict("index")
        )
    primary = rank_payload["mse_rare"]
    pass_by_method = {}
    for method in PRIMARY_METHODS:
        sorted_methods = sorted(primary.items(), key=lambda kv: (kv[1]["mean"], kv[1]["median"]))
        mean_rank = float(primary[method]["mean"])
        position = [name for name, _ in sorted_methods].index(method) + 1
        pass_by_method[method] = {
            "mean_rank": mean_rank,
            "position": position,
            "pass": bool(position <= 2 or mean_rank <= 2.5),
            "flagship_strong_pass": bool(mean_rank <= 2.5),
        }
    payload = {
        "rank_gate": "top-2 by primary metric or mean rank <= 2.5",
        "primary_metric": "mse_rare",
        "profile": profile,
        "method_count": int(df["method"].nunique()),
        "seed_count": int(df["seed"].nunique()),
        "dataset_count": int(df["dataset"].nunique()),
        "rank_summary": rank_payload,
        "dataset_ranks": dataset_ranks,
        "pass_by_method": pass_by_method,
        "any_pass": any(v["pass"] for v in pass_by_method.values()),
    }
    summary_path = output_dir / f"radical_{profile}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "radical_smoke_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TailMender experiments on tabular regression benchmarks.")
    parser.add_argument("--output-dir", default="results/tailmender")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=42, help="Single seed kept for backward compatibility.")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 42,43,44.")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names; default runs all nine benchmarks.")
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_csv_ints(args.seeds) if args.seeds else [args.seed]
    max_train, max_test = resolve_caps(args.profile, args.max_train, args.max_test)
    datasets = list(UAI9_DATASETS)
    if args.datasets:
        requested = [part.strip() for part in args.datasets.split(",") if part.strip()]
        unknown = sorted(set(requested) - set(UAI9_DATASETS))
        if unknown:
            raise ValueError(f"Unknown dataset(s): {unknown}. Available: {UAI9_DATASETS}")
        datasets = requested
    methods = [
        "No-Rebalance",
        "StaticTailWeight",
        "InverseDensityWeight",
        "LDSWeight",
        "RandomTailReplay",
        "SMOTER-Batch",
        "SMOGN-Batch",
        "REBAGG-Resample",
    ]
    if torch_available():
        methods += ["RankSim", "ConR", "VIR"]
    else:
        print("WARNING: torch unavailable; skipping deep baselines RankSim/ConR/VIR.")
    methods.append("TailMender")
    methods = tuple(methods)
    rows: list[dict] = []
    manifest: list[dict] = []

    print("=" * 72)
    print(f"TailMender {args.profile} test on nine tabular benchmarks")
    print("=" * 72)
    print(f"Methods: {', '.join(methods)}")
    print(f"Seeds: {seeds}")
    print(f"Caps: max_train={max_train} max_test={max_test}")
    print("Rank gate: top-2 by rare MSE or mean rank <= 2.5")

    for dataset in datasets:
        X, y, info = load_real_world_dataset(dataset)
        manifest.append(info)
        X_train, y_train, X_test, y_test = chronological_split_scale(X, y, 0.30)
        if max_train is not None or max_test is not None:
            X_train, y_train, X_test, y_test = cap_chronological(
                X_train,
                y_train,
                X_test,
                y_test,
                max_train if max_train is not None else len(y_train),
                max_test if max_test is not None else len(y_test),
            )
        thresholds = tail_thresholds(y_train, 0.10, 0.90)
        print(f"\n[{dataset}] train={len(y_train)} test={len(y_test)} thresholds={thresholds}")
        for seed in seeds:
            print(f"  seed={seed}")
            for method in methods:
                t0 = time.perf_counter()
                pred, stats = fit_predict_method(method, X_train, y_train, X_test, thresholds, seed, args.n_estimators)
                metrics = evaluate_prediction(y_test, pred, thresholds)
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "seconds": round(time.perf_counter() - t0, 3),
                    **metrics,
                    **stats,
                }
                rows.append(row)
                print(
                    f"    {method:<20} rare_mse={metrics['mse_rare']:.4g} "
                    f"rare_mae={metrics['mae_rare']:.4g} bulk_mae={metrics['mae_bulk']:.4g} "
                    f"mse={metrics['mse']:.4g}"
                )

    csv_path = output_dir / f"radical_{args.profile}_results.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.profile != "smoke":
        compat_csv_path = output_dir / "radical_smoke_results.csv"
        with compat_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = summarize_rank(rows, output_dir, args.profile)
    print("\n" + "=" * 72)
    print(f"Saved: {csv_path}")
    print(f"Saved: {output_dir / f'radical_{args.profile}_summary.json'}")
    print(f"PASS BY METHOD: {summary['pass_by_method']}")
    print(f"ANY PASS: {summary['any_pass']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
