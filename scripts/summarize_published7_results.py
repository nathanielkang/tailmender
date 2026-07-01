#!/usr/bin/env python3
"""Summarize TailMender full-run CSV for published-baseline-only reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PUBLISHED7_BASELINES = (
    "LDSWeight",
    "SMOTER-Batch",
    "SMOGN-Batch",
    "REBAGG-Resample",
    "RankSim",
    "ConR",
    "VIR",
)
PRIMARY_METHOD = "TailMender"
METRICS = ("mse_rare", "mae_rare", "mae_bulk", "mse", "mae")
RANK_METRICS = ("mse_rare", "mae_rare", "mae_bulk", "mse", "mae")


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = set(PUBLISHED7_BASELINES) | {PRIMARY_METHOD}
    missing = keep - set(df["method"].unique())
    if missing:
        raise ValueError(f"Missing methods in CSV: {sorted(missing)}")
    return df[df["method"].isin(keep)].copy()


def canonical_table(df: pd.DataFrame) -> dict:
    agg_cols = ["mse_rare", "mae_rare", "mse_bulk", "mae_bulk", "mse", "mae"]
    means = (
        df.groupby(["dataset", "method"], as_index=False)[agg_cols]
        .mean()
        .round(4)
    )
    out: dict = {}
    for dataset, grp in means.groupby("dataset"):
        out[dataset] = {
            rec["method"]: {c: float(rec[c]) for c in agg_cols}
            for rec in grp.to_dict(orient="records")
        }
    return out


def rank_summary(df: pd.DataFrame) -> dict:
    rank_payload: dict = {}
    dataset_ranks: dict = {}
    for metric in RANK_METRICS:
        means = df.groupby(["dataset", "method"], as_index=False)[metric].mean()
        means["rank"] = means.groupby("dataset")[metric].rank(method="average", ascending=True)
        dataset_ranks[metric] = means.sort_values(["dataset", "rank"]).to_dict("records")
        rank_payload[metric] = (
            means.groupby("method")["rank"]
            .agg(["mean", "median"])
            .sort_values("mean")
            .round(4)
            .to_dict("index")
        )
    primary = rank_payload["mse_rare"]
    sorted_methods = sorted(primary.items(), key=lambda kv: (kv[1]["mean"], kv[1]["median"]))
    tm = primary[PRIMARY_METHOD]
    position = [name for name, _ in sorted_methods].index(PRIMARY_METHOD) + 1
    pass_by_method = {
        PRIMARY_METHOD: {
            "mean_rank": float(tm["mean"]),
            "median_rank": float(tm["median"]),
            "position": position,
            "pass": bool(position <= 2 or tm["mean"] <= 2.5),
            "flagship_strong_pass": bool(tm["mean"] <= 2.5),
        }
    }
    return {
        "rank_gate": "top-2 by primary metric or mean rank <= 2.5",
        "primary_metric": "mse_rare",
        "filter": "published-baseline-only (7 external + TailMender)",
        "published_baselines": list(PUBLISHED7_BASELINES),
        "method_count": int(df["method"].nunique()),
        "seed_count": int(df["seed"].nunique()),
        "dataset_count": int(df["dataset"].nunique()),
        "rank_summary": rank_payload,
        "dataset_ranks": dataset_ranks,
        "pass_by_method": pass_by_method,
        "any_pass": pass_by_method[PRIMARY_METHOD]["pass"],
    }


def format_report(summary: dict, canonical: dict) -> str:
    lines = [
        "# TailMender Published-Baseline-Only Experiment Report",
        "",
        f"Filter: **published-baseline-only** — {len(PUBLISHED7_BASELINES)} external published baselines + TailMender.",
        "Controls/heuristics (No-Rebalance, StaticTailWeight, etc.) are excluded.",
        "",
        "## Mean ranks (lower is better)",
        "",
        "| Metric | " + " | ".join(
            m for m in summary["rank_summary"]["mse_rare"]
        ) + " |",
        "| --- | " + " | ".join(["---"] * len(summary["rank_summary"]["mse_rare"])) + " |",
    ]
    for metric in RANK_METRICS:
        row = summary["rank_summary"][metric]
        cells = [f"{row[m]['mean']:.2f}" for m in row]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    tm = summary["pass_by_method"][PRIMARY_METHOD]
    lines += [
        "",
        "## TailMender gate",
        "",
        f"- Primary metric: `{summary['primary_metric']}`",
        f"- Mean rank: **{tm['mean_rank']:.2f}** (position **{tm['position']}** of {summary['method_count']})",
        f"- Loose pass (top-2 or mean rank <= 2.5): **{'PASS' if tm['pass'] else 'FAIL'}**",
        f"- Flagship strong pass (mean rank <= 2.5): **{'PASS' if tm['flagship_strong_pass'] else 'FAIL'}**",
        "",
        "## Dataset-level rare-MSE ranks (TailMender)",
        "",
        "| Dataset | TailMender rank | Best method |",
        "| --- | ---: | --- |",
    ]
    rare_rows = summary["dataset_ranks"]["mse_rare"]
    by_ds: dict[str, list] = {}
    for r in rare_rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for dataset in sorted(by_ds):
        rows = sorted(by_ds[dataset], key=lambda x: x["rank"])
        tm_row = next((r for r in rows if r["method"] == PRIMARY_METHOD), None)
        best = rows[0]["method"] if rows else "?"
        rank = f"{tm_row['rank']:.0f}" if tm_row else "?"
        lines.append(f"| {dataset} | {rank} | {best} |")

    lines += ["", "## Published baselines included", ""]
    for b in PUBLISHED7_BASELINES:
        lines.append(f"- {b}")
    lines.append(f"- {PRIMARY_METHOD} (proposed)")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--csv-name", default="radical_full_results.csv")
    args = parser.parse_args()
    indir = Path(args.input_dir)
    csv_path = indir / args.csv_name
    if not csv_path.exists():
        alt = indir / "radical_smoke_results.csv"
        if alt.exists():
            csv_path = alt
        else:
            raise FileNotFoundError(csv_path)

    df = load_results(csv_path)
    canonical = canonical_table(df)
    summary = rank_summary(df)

    (indir / "canonical_published7_baseline_table.json").write_text(
        json.dumps(canonical, indent=2), encoding="utf-8"
    )
    (indir / "published7_rank_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (indir / "PUBLISHED7_EXPERIMENT_RESULT_REPORT.md").write_text(
        format_report(summary, canonical), encoding="utf-8"
    )
    # Also save filtered raw CSV
    df.to_csv(indir / "published7_filtered_results.csv", index=False)
    print(f"Wrote summaries to {indir}")
    print(f"TailMender pass: {summary['pass_by_method'][PRIMARY_METHOD]}")


if __name__ == "__main__":
    main()
