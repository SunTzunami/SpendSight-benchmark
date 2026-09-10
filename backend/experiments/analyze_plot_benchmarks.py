# backend/plot_benchmarks.py for plotting
"""
plot_results.py  (FIXED for expense_benchmark.py output format)

Reads the combined Excel output from expense_benchmark.py and generates
publication-quality plots. Works with both the combined file and
single/dual-mode files.

New column mapping vs old MLX script:
    OLD (MLX)           NEW (expense_benchmark)
    ─────────────────────────────────────────────
    Composite_Acc    →  Task_Acc_Validated
    Router_Correct   →  FSP_Validated        (tool routing binary)
    Fn_Correct       →  Correct_Ratio_Validated (full exact match)
    Param_Correct    →  AVC_Validated        (param value accuracy)
    Summary_Time_s   →  (no summarizer; filled with 0)
    TC_Category      →  TC_Group
    Repetition       →  Rep

Usage:
    # combined file (both modes)
    python plot_results.py --input benchmark_outputs/..._combined.xlsx --output figures/

    # single file only
    python plot_results.py --input benchmark_outputs/..._single.xlsx --output figures/

Fixes applied vs original:
    FIX-1  _pub_style() called once in generate_plots(); removed from every sub-function.
    FIX-2  _generate_model_accuracy_by_category() now plots one subplot per mode.
    FIX-3  _generate_model_bfcl_radar_comparison() adds one series per mode rather than collapsing modes.
    FIX-4  _generate_model_difficulty_breakdown_comparison() shows a breakdown by task complexity.
    FIX-5  _generate_error_taxonomy_plot() – removed unused variable i from outer loop.
    FIX-6  _generate_comparison_plots() – removed unused `import matplotlib.patches`.
    FIX-7  Category heatmap uses a grey bad-colour for NaN cells instead of blank white.
    FIX-8  Pareto front scatter uses ax.plot() with sorted frontier points and
           where="pre" step for a correct staircase direction.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import json
import re
import ast
from typing import Optional

# For Confusion Matrix
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None

# ──────────────────────────────────────────────────────────────────────────────
# PALETTE / STYLE
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]

_STAGE_COLORS = {
    "Router":     "#0072B2",
    "Specialist": "#E69F00",
}

_MODE_PALETTE = {
    "single": "#4C72B0",
    "dual":   "#C44E52",
}

_ERROR_COLORS = {
    "CORRECT":           "#2CA02C",
    "TOOL_WRONG":        "#D62728",
    "PARSE_FAILURE":     "#FF7F0E",
    "PARAM_MISSING":     "#9467BD",
    "PARAM_VALUE_WRONG": "#E377C2",
    "PARAM_EXTRA":       "#BCBD22",
    "FORMAT_VIOLATION":  "#17BECF",
}

# Map to new column names
_METRIC_STYLES = [
    ("Routing",    "avg_router_acc", "///",    "#0072B2"),
    ("Exact Match","avg_fn_acc",     "\\\\\\", "#E69F00"),
    ("Param Vals", "avg_param_acc",  "...",    "#009E73"),
]

ALLOWED_TOOLS = [
    "plot_time_series",
    "plot_distribution",
    "plot_comparison_bars",
    "calculate_total",
    "get_top_expenses",
]

SHORT_LABELS = {
    "plot_time_series":     "time_series",
    "plot_distribution":    "distribution",
    "plot_comparison_bars": "comparison",
    "calculate_total":      "calc_total",
    "get_top_expenses":     "top_expenses",
}



def _pub_style() -> None:
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "axes.grid":          True,
        "grid.linestyle":     "--",
        "grid.linewidth":     0.4,
        "grid.alpha":         0.5,
        "grid.color":         "#cccccc",
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth":    1.5,
        "lines.markersize":   6,
        "legend.frameon":     True,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#cccccc",
        "legend.handlelength":1.5,
    })


def _short(model_id: str) -> str:
    name = str(model_id).split("/")[-1]
    for pat in ("-MLX-", "-MLX", "-Instruct", "-instruct", ".gguf", ".GGUF"):
        name = name.replace(pat, "")
    # strip GGUF quantization suffixes like -Q4_K_M, -Q8_0, -IQ3_XS etc.
    name = re.sub(r"-[QI][\dA-Z_]+$", "", name, flags=re.IGNORECASE)
    
    # Custom mappings for clean presentation in figures:
    name_map = {
        "minicpm5-1b": "MiniCPM 1B",
        "gemma-3-1b-it": "Gemma-3 1B",
        "EXAONE-4.0-1.2B": "EXAONE 1.2B",
        "LFM2.5-1.2B": "LFM2.5 Instruct 1.2B",
        "LFM2-1.2B": "LFM2 1.2B",
        "Qwen3.5-0.8B": "Qwen3.5 0.8B",
        "Qwen3.5-2B": "Qwen3.5 2B",
        "Qwen3.5-4B-UD": "Qwen3.5 4B",
        "Qwen3.5-4B": "Qwen3.5 4B",
        "qwen3.5-4b-ud-iq2_m": "Qwen3.5 4B",
        "ai9stars_G9v3-3B": "G9v3 3B",
        "g9v3-3b": "G9v3 3B",
    }
    return name_map.get(name, name)


def _model_sort_key(model_id: str):
    raw_name = str(model_id).lower()
    short_name = _short(model_id).lower()
    
    # G9v3 3B reference model: placed at the very end
    if "g9v3" in raw_name or "ai9stars" in raw_name or "3b" in short_name:
        return (99, 3.0, raw_name)
    
    # Sub-2B models in standard order by family / size:
    if "exaone" in raw_name:
        return (1, 1.2, raw_name)
    elif "gemma" in raw_name:
        return (2, 1.0, raw_name)
    elif "lfm" in raw_name:
        return (3, 1.2, raw_name)
    elif "minicpm" in raw_name:
        return (4, 1.0, raw_name)
    elif "qwen" in raw_name:
        if "0.8b" in raw_name or "0.8b" in short_name:
            return (5, 0.8, raw_name)
        elif "2b" in raw_name or "2b" in short_name:
            return (6, 2.0, raw_name)
        elif "4b" in raw_name or "4b" in short_name:
            return (7, 4.0, raw_name)
        return (8, 99.0, raw_name)
    
    return (50, 0, raw_name)


def sort_models(model_list):
    return sorted(list(model_list), key=_model_sort_key)


def _savefig(fig, out_dir: str, name: str, fmt: str = "png") -> None:
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING — maps new column names to internal names used by plot functions
# ──────────────────────────────────────────────────────────────────────────────

# New column names produced by expense_benchmark.py
_NEW_COLS = {
    "Model",
    "TC_ID", "TC_Group", "TC_Difficulty", "Rep",
    "Total_Time_s", "Router_Time_s", "Specialist_Time_s",
    "Benchmark_Mode",
    "Task_Acc_Validated",
    "FSP_Validated",
    "Correct_Ratio_Validated",
    "AVC_Validated",
    "ACS_Validated",
    "Task_Acc_Raw",
    "FSP_Raw",
    "Error_Type_Raw", "Error_Type_Validated",
    "Prompt_Tokens", "Completion_Tokens",
}


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    print(f"Reading '{file_path}' ...")
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        try:
            df = pd.read_excel(file_path, sheet_name="Raw")
        except Exception:
            try:
                df = pd.read_excel(file_path, sheet_name="Raw Observations")
            except Exception:
                df = pd.read_excel(file_path, sheet_name=0)

    present = set(df.columns)

    # ── Detect format ────────────────────────────────────────────────────────
    if "Task_Acc_Validated" in present:
        # New expense_benchmark.py format
        print("  Detected: expense_benchmark.py output format")
        df = df.rename(columns={
            "TC_Group": "TC_Category",
            "Rep":      "Repetition",
        })
        # Create unified accuracy / component columns matching old plot logic
        df["Composite_Acc"] = df["Task_Acc_Validated"]
        df["Router_Correct"] = df["FSP_Validated"]           # 1.0 if correct tool
        df["Fn_Correct"]     = df["Correct_Ratio_Validated"] # 1.0 if fully correct
        df["Param_Correct"]  = df["AVC_Validated"]           # 0‒1 param value score
        # No summarizer stage in this pipeline
        df["Summary_Time_s"] = 0.0

    elif "Composite_Acc" in present:
        # Old MLX format — convert OK/FAIL strings
        print("  Detected: legacy MLX output format")
        df = df.rename(columns={
            "Category":   "TC_Category",
            "Total (s)":  "Total_Time_s",
            "Router (s)": "Router_Time_s",
            "Spec (s)":   "Specialist_Time_s",
            "Summary (s)":"Summary_Time_s",
            "Rep":        "Repetition",
        })
        for col in ["Router_Correct", "Fn_Correct", "Param_Correct"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: 1.0 if str(x).strip().upper() == "OK"
                    else (0.0 if str(x).strip().upper() == "FAIL" else x)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if "Summary_Time_s" not in df.columns:
            df["Summary_Time_s"] = 0.0

    else:
        raise ValueError(
            f"Unrecognised Excel format. Found columns: {list(present)[:20]}"
        )

    # Ensure numeric
    for col in ["Total_Time_s", "Router_Time_s", "Specialist_Time_s",
                "Summary_Time_s", "Composite_Acc",
                "Router_Correct", "Fn_Correct", "Param_Correct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Benchmark_Mode column (may not exist in old format)
    if "Benchmark_Mode" not in df.columns:
        df["Benchmark_Mode"] = "single"

    print(f"  Loaded {len(df)} rows | "
          f"modes: {sorted(df['Benchmark_Mode'].unique())} | "
          f"models: {df['Model'].nunique()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# PLOTTING
# FIX-1: _pub_style() called once here; removed from every sub-function below.
# ──────────────────────────────────────────────────────────────────────────────

def generate_plots(df: pd.DataFrame, out_dir: str, mode_filter: Optional[str] = None) -> None:
    # Ensure output directory exists before generating plots
    os.makedirs(out_dir, exist_ok=True)
    
    # FIX-1: single call here; sub-functions no longer call _pub_style().
    _pub_style()

    modes = df["Benchmark_Mode"].unique().tolist()
    if mode_filter and mode_filter in modes:
        modes = [mode_filter]

    for mode in modes:
        mdf = df[df["Benchmark_Mode"] == mode].copy()
        if mdf.empty:
            continue
        prefix = f"{mode}_"
        print(f"\n── Generating plots for mode='{mode}' ({len(mdf)} rows) ──")
        _generate_mode_plots(mdf, out_dir, prefix)
        _generate_bfcl_radar(mdf, out_dir, prefix)
        _generate_confusion_matrix(mdf, out_dir, prefix)

    # Global plots (all modes)
    _generate_tradeoff_scatter(df, out_dir)

    # Validation analysis — raw vs validated, works on full df (all modes together)
    _generate_validation_plots(df, out_dir)

    # New analysis plots from hardened benchmark
    _generate_error_taxonomy_plot(df, out_dir)
    _generate_difficulty_breakdown(df, out_dir)
    _generate_token_efficiency_plot(df, out_dir)

    # Combined comparison plots if both modes are present
    if len(df["Benchmark_Mode"].unique()) > 1:
        print("\n── Generating combined comparison plots ──")
        _generate_comparison_plots(df, out_dir)
        _generate_combined_bfcl_radar(df, out_dir)
        _generate_model_radar_grid(df, out_dir)

        print("\n── Generating model-level comparison plots ──")
        _generate_model_accuracy_by_category(df, out_dir)
        _generate_model_bfcl_radar_comparison(df, out_dir)
        _generate_model_difficulty_breakdown_comparison(df, out_dir)


def _generate_mode_plots(df: pd.DataFrame, out_dir: str, prefix: str) -> None:
    model_col  = sort_models(df["Model"].unique())
    color_map  = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(model_col)}
    marker_map = {m: ["o","s","^","D","v","p","P","*","X"][i % 9] for i, m in enumerate(model_col)}
    labels     = [_short(m) for m in model_col]
    rng        = np.random.default_rng(42)

    # ── Aggregations ────────────────────────────────────────────────────────
    mg = df.groupby("Model").agg(
        avg_total_s    =("Total_Time_s",       "mean"),
        std_total_s    =("Total_Time_s",       "std"),
        avg_router_s   =("Router_Time_s",      "mean"),
        avg_spec_s     =("Specialist_Time_s",  "mean"),
        avg_sum_s      =("Summary_Time_s",     "mean"),
        avg_composite  =("Composite_Acc",      "mean"),
        avg_router_acc =("Router_Correct",     "mean"),
        avg_fn_acc     =("Fn_Correct",         "mean"),
        avg_param_acc  =("Param_Correct",      "mean"),
    ).reindex(model_col)
    # std can be NaN if only 1 rep — replace with 0
    mg["std_total_s"] = mg["std_total_s"].fillna(0.0)

    # ── Define stages for latency plots and summary panel ───────────────────
    show_summarizer = mg["avg_sum_s"].sum() > 0
    stages = [("Router", "avg_router_s", _STAGE_COLORS["Router"]),
              ("Specialist", "avg_spec_s", _STAGE_COLORS["Specialist"])]
    if show_summarizer:
        stages.append(("Summarizer", "avg_sum_s", "#009E73"))

    cat_pivot = (
        df.groupby(["Model", "TC_Category"])["Composite_Acc"]
        .mean()
        .unstack("TC_Category")
        .reindex(model_col)
        * 100
    )
    categories = cat_pivot.columns.tolist()

    tc_avg = (
        df.groupby(["Model", "TC_ID"])["Composite_Acc"]
        .mean()
        .reset_index()
    )

    jitter_w = 0.12

    # 1. Accuracy vs Latency ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for model, row in mg.iterrows():
        ax.errorbar(
            row["avg_total_s"], row["avg_composite"] * 100,
            xerr=row["std_total_s"],
            fmt=marker_map[model], color=color_map[model],
            markersize=9, alpha=0.8,
            capsize=3, capthick=1.0, elinewidth=1.0,
            label=_short(model), zorder=3,
        )
    ax.set_xlabel("Average Total Latency (s)")
    ax.set_ylabel("Task Accuracy (%)")
    ax.set_title("Accuracy vs. Latency")
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}acc_vs_latency")

    # 2. Latency Breakdown ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(3.5, len(labels) * 0.9 + 1.5), 3.5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for stage_label, col_key, color in stages:
        vals = mg[col_key].values
        ax.bar(x, vals, 0.55, bottom=bottom, label=stage_label,
               color=color, edgecolor="white", linewidth=0.5, zorder=3)
        bottom += vals
    ax.errorbar(x, mg["avg_total_s"].values, yerr=mg["std_total_s"].values,
                fmt="none", color="#333333", capsize=3, capthick=0.8, elinewidth=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Pipeline Latency Breakdown")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}latency_breakdown")

    # 3. Accuracy Breakdown ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(3.5, len(labels) * 1.2 + 1.5), 3.5))
    x = np.arange(len(labels))
    w = 0.22
    offs = np.array([-w, 0, w])
    for (lbl, key, hatch, color), offset in zip(_METRIC_STYLES, offs):
        vals = mg[key].values * 100
        bars = ax.bar(x + offset, vals, w, label=lbl, color=color,
                      alpha=0.85, hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Accuracy by Component\n(Routing FSP | Exact Match | Param Value AVC)")
    ax.set_ylim(0, 130)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.axhline(100, color="#cccccc", ls="--", lw=0.6, zorder=1)
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}accuracy_breakdown")

    # 4. Category Heatmap ────────────────────────────────────────────────────
    # FIX-7: use a grey bad-colour so NaN cells are clearly marked, not blank white.
    fig, ax = plt.subplots(figsize=(max(5.5, len(categories) * 1.1 + 1.5),
                                    max(3.5, len(labels) * 0.7 + 1.0)))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(cat_pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    for i in range(len(labels)):
        for j in range(len(categories)):
            v = cat_pivot.values[i, j]
            if not np.isnan(v):
                tc_c = "white" if v < 40 or v > 80 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=11, color=tc_c, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Task Accuracy (%)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("Task Accuracy by Model × Task Category", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}category_heatmap")

    # 5. Per-TC Strip Plot ───────────────────────────────────────────────────
    bp_data = [tc_avg[tc_avg["Model"] == m]["Composite_Acc"].values * 100 for m in model_col]
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.1 + 1), 3.5))
    bp = ax.boxplot(bp_data, positions=np.arange(len(model_col)), widths=0.3,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    boxprops=dict(linewidth=0.8))
    for patch, model in zip(bp["boxes"], model_col):
        patch.set_facecolor(color_map[model])
        patch.set_alpha(0.25)
    for pos, model in zip(np.arange(len(model_col)), model_col):
        vals   = tc_avg[tc_avg["Model"] == model]["Composite_Acc"].values * 100
        jitter = rng.uniform(-jitter_w, jitter_w, len(vals))
        ax.scatter(pos + jitter, vals, s=18, color=color_map[model],
                   alpha=0.75, zorder=4, linewidths=0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Task Accuracy (%) — per TC")
    ax.set_title("Per-Question Accuracy Distribution")
    ax.set_ylim(-5, 108)
    ax.axhline(100, color="#cccccc", ls="--", lw=0.6)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}per_tc_strip")

    # 6. Latency Box Plot ────────────────────────────────────────────────────
    lat_data = [df[df["Model"] == m]["Total_Time_s"].values for m in model_col]
    fig, ax  = plt.subplots(figsize=(max(4, len(labels) * 1.1 + 1), 3.5))
    bp2 = ax.boxplot(lat_data, positions=np.arange(len(model_col)), widths=0.4,
                     patch_artist=True, showfliers=True,
                     flierprops=dict(marker=".", markersize=3, alpha=0.4),
                     medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8),
                     boxprops=dict(linewidth=0.8))
    for patch, model in zip(bp2["boxes"], model_col):
        patch.set_facecolor(color_map[model])
        patch.set_alpha(0.55)
    ax.set_xticks(np.arange(len(model_col)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Total Latency (s)")
    ax.set_title("Total Latency Distribution")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}latency_box")

    # 7. Summary Panel ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 6.5))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_lat_brk = fig.add_subplot(gs[0, 1])
    ax_acc_brk = fig.add_subplot(gs[0, 2])
    ax_heatmap = fig.add_subplot(gs[1, 0:2])
    ax_strip   = fig.add_subplot(gs[1, 2])

    # (a) scatter
    for model, row in mg.iterrows():
        ax_scatter.errorbar(
            row["avg_total_s"], row["avg_composite"] * 100,
            xerr=row["std_total_s"],
            fmt=marker_map[model], color=color_map[model],
            markersize=8, alpha=0.8, capsize=2.5, capthick=0.8,
            elinewidth=0.8, label=_short(model), zorder=3,
        )
    ax_scatter.set_xlabel("Avg Latency (s)")
    ax_scatter.set_ylabel("Task Acc. (%)")
    ax_scatter.set_ylim(0, 105)
    ax_scatter.set_xlim(left=0)
    ax_scatter.yaxis.set_major_locator(MultipleLocator(20))
    ax_scatter.legend(loc="lower right", fontsize=6)

    # (b) latency breakdown
    x_b = np.arange(len(labels))
    bot = np.zeros(len(labels))
    for stage_label, col_key, color in stages:
        vals = mg[col_key].values
        ax_lat_brk.bar(x_b, vals, 0.5, bottom=bot, label=stage_label,
                       color=color, edgecolor="white", linewidth=0.4, zorder=3)
        bot += vals
    ax_lat_brk.set_xticks(x_b)
    ax_lat_brk.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    ax_lat_brk.set_ylabel("Latency (s)")
    ax_lat_brk.legend(fontsize=6, loc="upper right")
    ax_lat_brk.set_ylim(bottom=0)

    # (c) accuracy breakdown
    x_c  = np.arange(len(labels))
    offs = np.array([-w, 0, w])
    for (lbl, key, hatch, color), off in zip(_METRIC_STYLES, offs):
        ax_acc_brk.bar(x_c + off, mg[key].values * 100, w, label=lbl,
                       color=color, alpha=0.85, hatch=hatch,
                       edgecolor="white", linewidth=0.3, zorder=3)
    ax_acc_brk.set_xticks(x_c)
    ax_acc_brk.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    ax_acc_brk.set_ylabel("Score (%)")
    ax_acc_brk.set_ylim(0, 135)
    ax_acc_brk.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.05, 1.0))
    ax_acc_brk.axhline(100, color="#cccccc", ls="--", lw=0.5)
    ax_acc_brk.yaxis.set_major_locator(MultipleLocator(25))

    # (d) heatmap — FIX-7 applied here too
    cmap_panel = plt.cm.RdYlGn.copy()
    cmap_panel.set_bad("#dddddd")
    im2 = ax_heatmap.imshow(cat_pivot.values, aspect="auto",
                             cmap=cmap_panel, vmin=0, vmax=100)
    ax_heatmap.set_xticks(np.arange(len(categories)))
    ax_heatmap.set_yticks(np.arange(len(labels)))
    ax_heatmap.set_xticklabels(categories, rotation=28, ha="right", fontsize=7)
    ax_heatmap.set_yticklabels(labels, fontsize=7)
    ax_heatmap.tick_params(length=0)
    ax_heatmap.spines[:].set_visible(False)
    for i in range(len(labels)):
        for j in range(len(categories)):
            v = cat_pivot.values[i, j]
            if not np.isnan(v):
                tc_c = "white" if v < 40 or v > 80 else "black"
                ax_heatmap.text(j, i, f"{v:.0f}", ha="center", va="center",
                                fontsize=6.5, color=tc_c, fontweight="bold")
    cbar2 = fig.colorbar(im2, ax=ax_heatmap, fraction=0.025, pad=0.02)
    cbar2.set_label("Acc. (%)", fontsize=7)
    cbar2.ax.tick_params(labelsize=6)

    # (e) strip
    bp3 = ax_strip.boxplot(bp_data, positions=np.arange(len(model_col)), widths=0.3,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", linewidth=1.2),
                           whiskerprops=dict(linewidth=0.7),
                           capprops=dict(linewidth=0.7),
                           boxprops=dict(linewidth=0.7))
    for patch, model in zip(bp3["boxes"], model_col):
        patch.set_facecolor(color_map[model])
        patch.set_alpha(0.25)
    for pos, model in zip(np.arange(len(model_col)), model_col):
        vals   = tc_avg[tc_avg["Model"] == model]["Composite_Acc"].values * 100
        jitter = rng.uniform(-jitter_w, jitter_w, len(vals))
        ax_strip.scatter(pos + jitter, vals, s=14, color=color_map[model],
                         alpha=0.75, zorder=4, linewidths=0)
    ax_strip.set_xticks(np.arange(len(labels)))
    ax_strip.set_xticklabels(labels, rotation=22, ha="right", fontsize=7)
    ax_strip.set_ylabel("Acc. (%) per TC")
    ax_strip.set_ylim(-5, 108)
    ax_strip.axhline(100, color="#cccccc", ls="--", lw=0.5)
    ax_strip.yaxis.set_major_locator(MultipleLocator(25))

    mode_label = prefix.replace("_", "").capitalize()
    for ax_p, plabel in zip(
        [ax_scatter, ax_lat_brk, ax_acc_brk, ax_heatmap, ax_strip],
        ["(a)", "(b)", "(c)", "(d)", "(e)"],
    ):
        ax_p.set_title(plabel, loc="left", fontweight="bold", fontsize=9, pad=2)

    fig.suptitle(
        f"SpendSight Benchmark — {mode_label} Architecture",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}summary_panel")


def _generate_bfcl_radar(df: pd.DataFrame, out_dir: str, prefix: str) -> None:
    """BFCL-style radar footprint."""
    pivot = df.pivot_table(
        index="Model", columns="TC_Category",
        values="Composite_Acc", aggfunc="mean",
    )
    categories = [str(c).replace("_", " ").title() for c in pivot.columns]
    N = len(categories)
    if N < 3:
        print(f"  [skip] Radar for {prefix} (need ≥3 categories)")
        return

    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=11, weight="bold")
    ax.tick_params(axis="x", pad=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#444444", size=10.5)
    plt.ylim(0, 1.05)

    model_ids = pivot.index.tolist()
    for i, model in enumerate(model_ids):
        color = _PALETTE[i % len(_PALETTE)]
        vals = pivot.loc[model].values.flatten().tolist()
        vals += [vals[0]]
        ax.plot(angles, vals, linewidth=1.5, linestyle="solid",
                label=_short(model), color=color, alpha=0.9)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
               title="Models", frameon=True, fontsize=10.5, title_fontsize=11)
    mode_label = prefix.replace("_", "").capitalize()
    ax.set_title(f"Task Footprint — {mode_label} Mode\n(BFCL Radar Style)",
              size=13, weight="bold", y=1.1)

    _savefig(fig, out_dir, f"{prefix}bfcl_radar")


def _generate_confusion_matrix(df: pd.DataFrame, out_dir: str, prefix: str) -> None:
    """Routing confusion matrix."""
    if confusion_matrix is None:
        print("  [skip] sklearn not installed — skipping confusion matrix.")
        return

    if "Expected_Tool" not in df.columns or "Pred_Tool_Validated" not in df.columns:
        print("  [skip] Routing columns missing — skipping confusion matrix.")
        return

    short_names = [SHORT_LABELS.get(t, t) for t in ALLOWED_TOOLS]

    y_true = df["Expected_Tool"].fillna("unknown").astype(str)
    y_pred = df["Pred_Tool_Validated"].fillna("unknown").astype(str)

    cm = confusion_matrix(y_true, y_pred, labels=ALLOWED_TOOLS)

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                           cm / cm.sum(axis=1, keepdims=True), 0.0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", size=11, weight="bold")
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Predicted Tool", fontsize=11, weight="bold")
    ax.set_ylabel("True Tool", fontsize=11, weight="bold")

    mode_label = prefix.replace("_", "").capitalize()
    ax.set_title(f"Tool Routing Confusion — {mode_label} Mode", fontsize=12, weight="bold", pad=12)

    for i in range(len(ALLOWED_TOOLS)):
        for j in range(len(ALLOWED_TOOLS)):
            val   = cm_norm[i, j]
            count = cm[i, j]
            if count > 0:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}\n({count})",
                        ha="center", va="center", fontsize=9, color=color, fontweight="semibold")

    fig.tight_layout()
    _savefig(fig, out_dir, f"{prefix}confusion_matrix")


def _generate_tradeoff_scatter(df: pd.DataFrame, out_dir: str) -> None:
    """Latency vs Accuracy Pareto tradeoff.

    FIX-8: Pareto front is built correctly and drawn with ax.plot() using
    sorted frontier points instead of ax.step(where='post'), which was
    connecting non-adjacent points into a misleading staircase.
    """
    agg = df.groupby(["Model", "Benchmark_Mode"]).agg(
        avg_time=("Total_Time_s", "mean"),
        avg_acc=("Composite_Acc", "mean")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    modes   = sorted(agg["Benchmark_Mode"].unique())
    palette = {"single": "#0072B2", "dual": "#D55E00"}
    markers = {"single": "o", "dual": "s"}

    for mode in modes:
        mdf = agg[agg["Benchmark_Mode"] == mode]
        ax.scatter(mdf["avg_time"], mdf["avg_acc"] * 100,
                   label=f"{mode.capitalize()} Mode",
                   color=palette.get(mode, "#555555"),
                   marker=markers.get(mode, "D"),
                   s=80, alpha=0.8, edgecolor="white", linewidth=0.8, zorder=3)

        for _, row in mdf.iterrows():
            model_name = _short(row["Model"])
            dx, dy = 0.12, 0.6
            ha, va = "left", "bottom"
            
            # Prevent label overlap or obscuring
            if mode == "dual" and model_name == "Gemma-3 1B":
                dx, dy = -0.12, -1.8
                ha, va = "right", "top"
            elif mode == "dual" and model_name == "LFM2.5 Instruct 1.2B":
                dx, dy = 0.12, 1.2
                ha, va = "left", "bottom"

            ax.text(row["avg_time"] + dx, row["avg_acc"] * 100 + dy,
                    model_name, fontsize=10.5, fontweight="bold", alpha=0.9, zorder=5,
                    ha=ha, va=va)

    # FIX-8: proper Pareto front — sort by latency, keep only points that
    # improve accuracy; then draw as a plain line (not step).
    pareto_sorted = agg.sort_values("avg_time")
    front, max_acc = [], -1.0
    for _, row in pareto_sorted.iterrows():
        if row["avg_acc"] > max_acc:
            front.append(row)
            max_acc = row["avg_acc"]

    if front:
        fdf = pd.DataFrame(front).sort_values("avg_time")
        ax.plot(fdf["avg_time"], fdf["avg_acc"] * 100,
                color="#888888", linestyle="--", linewidth=1.0, alpha=0.5,
                label="Pareto Front", zorder=1)

    ax.set_xlabel("Average Latency (s)", fontsize=12)
    ax.set_ylabel("Task Accuracy (%)", fontsize=12)
    ax.set_title("Latency vs Accuracy Tradeoff", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10.5, loc="center right", bbox_to_anchor=(1.02, 0.38))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir, "tradeoff_pareto")


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION ANALYSIS PLOTS
# ──────────────────────────────────────────────────────────────────────────────

_VAL_RAW_COLS = [
    "Task_Acc_Raw", "Task_Acc_Validated",
    "FSP_Raw",      "FSP_Validated",
    "ACS_Raw",      "ACS_Validated",
    "AVC_Raw",      "AVC_Validated",
]

_METRIC_PAIRS = [
    ("Task Acc",  "Task_Acc_Raw", "Task_Acc_Validated", "Overall Task Accuracy (F1)"),
    ("FSP",       "FSP_Raw",      "FSP_Validated",      "Function Selection (Tool Routing)"),
    ("ACS",       "ACS_Raw",      "ACS_Validated",      "Arg Coverage Score (F1 on arg names)"),
    ("AVC",       "AVC_Raw",      "AVC_Validated",      "Arg Value Correctness"),
]

_VAL_COLORS = {
    "Raw":       "#9EA0A3",
    "Validated": "#2CA02C",
}


def _has_validation_cols(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in _VAL_RAW_COLS)


def _generate_validation_plots(df: pd.DataFrame, out_dir: str) -> None:
    if not _has_validation_cols(df):
        print("  [skip] Validation columns not found — skipping validation plots.")
        return

    modes = sorted(df["Benchmark_Mode"].unique())
    if not modes:
        print("  [skip] No modes found for validation plots.")
        return

    models = sort_models(df["Model"].unique())
    if not models:
        print("  [skip] No models found for validation plots.")
        return

    print("\n── Generating validation analysis plots (LLM modes only) ──")
    n_metrics = len(_METRIC_PAIRS)
    n_models  = len(models)
    fig, axes = plt.subplots(
        len(modes), n_metrics,
        figsize=(n_metrics * max(3.0, n_models * 0.7 + 0.8), len(modes) * 3.5),
        sharey=False,
        squeeze=False,
    )

    for row_idx, mode in enumerate(modes):
        mdf   = df[df["Benchmark_Mode"] == mode]
        xlabs = [_short(m) for m in models]
        x     = np.arange(len(models))
        w     = 0.32

        for col_idx, (short_lbl, raw_col, val_col, desc) in enumerate(_METRIC_PAIRS):
            ax = axes[row_idx][col_idx]

            raw_means = [mdf[mdf["Model"] == m][raw_col].mean() * 100 for m in models]
            val_means = [mdf[mdf["Model"] == m][val_col].mean() * 100 for m in models]

            b1 = ax.bar(x - w / 2, raw_means, w,
                        label="Raw",       color=_VAL_COLORS["Raw"],
                        edgecolor="white", linewidth=0.5, zorder=3)
            b2 = ax.bar(x + w / 2, val_means, w,
                        label="Validated", color=_VAL_COLORS["Validated"],
                        edgecolor="white", linewidth=0.5, zorder=3)

            for xi, (rv, vv) in enumerate(zip(raw_means, val_means)):
                delta = vv - rv
                sign  = "+" if delta >= 0 else ""
                color = "#2CA02C" if delta > 0 else ("#C44E52" if delta < 0 else "#888888")
                ax.text(xi, max(rv, vv) + 1.5, f"{sign}{delta:.1f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=color, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(xlabs, rotation=20, ha="right", fontsize=7)
            ax.set_ylim(0, 118)
            ax.axhline(100, color="#cccccc", ls="--", lw=0.5)
            ax.yaxis.set_major_locator(MultipleLocator(20))
            ax.set_ylabel("Score (%)" if col_idx == 0 else "")

            title_mode = f"[{mode}] " if row_idx == 0 else ""
            ax.set_title(f"{title_mode}{short_lbl}\n{desc}", fontsize=7.5, pad=3)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="upper left", fontsize=7,
                          handles=[b1, b2], labels=["Raw LLM", "Validated"])

    fig.suptitle("Validation Lift — Raw LLM Output vs After Validation Step",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "validation_lift_bars")

    # ── V2: Signed delta heatmap ──────────────────────────────────────────────
    delta_rows = []
    for mode in modes:
        mdf = df[df["Benchmark_Mode"] == mode]
        for model in models:
            sub  = mdf[mdf["Model"] == model]
            row  = {"Label": f"{_short(model)}\n({mode})"}
            for short_lbl, raw_col, val_col, _ in _METRIC_PAIRS:
                row[short_lbl] = (sub[val_col].mean() - sub[raw_col].mean()) * 100
            delta_rows.append(row)

    delta_df     = pd.DataFrame(delta_rows).set_index("Label")
    metric_labels = [p[0] for p in _METRIC_PAIRS]

    fig, ax = plt.subplots(figsize=(len(metric_labels) * 1.6 + 1.6,
                                    len(delta_rows) * 0.85 + 1.5))
    vmax = max(abs(delta_df.values.max()), abs(delta_df.values.min()), 5)
    im   = ax.imshow(delta_df.values, aspect="auto",
                     cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(delta_rows)))
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(delta_df.index.tolist(), fontsize=10)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)

    for i in range(len(delta_rows)):
        for j in range(len(metric_labels)):
            v    = delta_df.values[i, j]
            sign = "+" if v >= 0 else ""
            tc_c = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{sign}{v:.1f}",
                    ha="center", va="center", fontsize=11,
                    color=tc_c, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Validated − Raw (pp)", fontsize=11, weight="bold")
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("Validation Δ Heatmap\n(positive = validation helped, negative = hurt)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    _savefig(fig, out_dir, "validation_delta_heatmap")

    # ── V3: Per-category validation lift ─────────────────────────────────────
    cats   = sorted(df["TC_Category"].unique())
    n_cats = len(cats)
    x_c    = np.arange(n_cats)
    w_c    = 0.32

    fig, axes3 = plt.subplots(
        1, len(modes),
        figsize=(max(5, n_cats * 0.9 + 2) * len(modes), 3.8),
        sharey=True,
    )
    if len(modes) == 1:
        axes3 = [axes3]

    for ax3, mode in zip(axes3, modes):
        mdf   = df[df["Benchmark_Mode"] == mode]
        raw_v = [mdf[mdf["TC_Category"] == c]["Task_Acc_Raw"].mean() * 100 for c in cats]
        val_v = [mdf[mdf["TC_Category"] == c]["Task_Acc_Validated"].mean() * 100 for c in cats]

        b1 = ax3.bar(x_c - w_c / 2, raw_v, w_c,
                     color=_VAL_COLORS["Raw"],       label="Raw",
                     edgecolor="white", linewidth=0.5, zorder=3)
        b2 = ax3.bar(x_c + w_c / 2, val_v, w_c,
                     color=_VAL_COLORS["Validated"],  label="Validated",
                     edgecolor="white", linewidth=0.5, zorder=3)

        for xi, (rv, vv) in enumerate(zip(raw_v, val_v)):
            delta = vv - rv
            if abs(delta) > 0.5:
                sign  = "+" if delta >= 0 else ""
                color = "#2CA02C" if delta > 0 else "#C44E52"
                ax3.text(xi, max(rv, vv) + 1.2, f"{sign}{delta:.0f}",
                         ha="center", va="bottom", fontsize=6.5,
                         color=color, fontweight="bold")

        ax3.set_xticks(x_c)
        ax3.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("Task Accuracy (%)")
        ax3.set_ylim(0, 120)
        ax3.axhline(100, color="#cccccc", ls="--", lw=0.5)
        ax3.yaxis.set_major_locator(MultipleLocator(20))
        ax3.set_title(f"{mode.capitalize()} Mode — Validation Lift per Task Category",
                      fontsize=9, fontweight="bold")
        ax3.legend(handles=[b1, b2], labels=["Raw LLM", "Validated"],
                   loc="lower right", fontsize=7)

    fig.suptitle("Task Accuracy: Raw vs Validated by Category",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "validation_per_category")

    # ── V4: Scatter — raw vs validated per observation ───────────────────────
    fig, ax4 = plt.subplots(figsize=(4.5, 4.5))

    for mode in modes:
        mdf = df[df["Benchmark_Mode"] == mode]
        col = _MODE_PALETTE.get(mode, "#555555")
        ax4.scatter(
            mdf["Task_Acc_Raw"] * 100,
            mdf["Task_Acc_Validated"] * 100,
            c=col, alpha=0.45, s=18, linewidths=0,
            label=mode.capitalize(), zorder=3,
        )

    ax4.plot([0, 100], [0, 100], color="#aaaaaa", lw=1.0, ls="--",
             zorder=1, label="No change (y = x)")

    ax4.fill_between([0, 100], [0, 100], [100, 100], alpha=0.04, color="#2CA02C")
    ax4.fill_between([0, 100], [0,   0], [100, 100], alpha=0.04, color="#C44E52")

    ax4.text(2, 97,  "Validation helped ▲", fontsize=7, color="#2CA02C", va="top")
    ax4.text(60, 3,  "Validation hurt ▼",   fontsize=7, color="#C44E52", va="bottom")

    ax4.set_xlabel("Raw Task Accuracy (%)")
    ax4.set_ylabel("Validated Task Accuracy (%)")
    ax4.set_title("Per-Observation: Raw vs Validated\n(each dot = one test case run)",
                  fontsize=9, fontweight="bold")
    ax4.set_xlim(-2, 105)
    ax4.set_ylim(-2, 105)
    ax4.set_aspect("equal")
    ax4.yaxis.set_major_locator(MultipleLocator(20))
    ax4.xaxis.set_major_locator(MultipleLocator(20))
    ax4.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    _savefig(fig, out_dir, "validation_scatter")


def _generate_error_taxonomy_plot(df: pd.DataFrame, out_dir: str) -> None:
    """Stacked bar plot of validated error types per model and mode."""
    if "Error_Type_Validated" not in df.columns:
        return

    import matplotlib.patheffects as path_effects

    modes = sorted(df["Benchmark_Mode"].unique())
    plot_models = sort_models(df["Model"].unique())

    if not modes:
        return

    fig, axes = plt.subplots(
        1, len(modes),
        figsize=(max(7, len(plot_models) * 1.2 + 1) * len(modes), 5),
        sharey=True,
    )
    if len(modes) == 1:
        axes = [axes]

    error_types = list(_ERROR_COLORS.keys())

    for _, (ax, mode) in enumerate(zip(axes, modes)):
        mdf = df[df["Benchmark_Mode"] == mode]
        if mdf.empty:
            continue

        x       = np.arange(len(plot_models))
        bottoms = np.zeros(len(plot_models))

        counts   = mdf.groupby(["Model", "Error_Type_Validated"]).size().unstack(fill_value=0)
        percents = (counts.T / counts.sum(axis=1)).T * 100

        for err_type in error_types:
            vals = []
            for m in plot_models:
                if m in percents.index:
                    vals.append(percents.loc[m, err_type] if err_type in percents.columns else 0)
                else:
                    vals.append(0)

            vals = np.array(vals)
            ax.bar(x, vals, bottom=bottoms, label=err_type.replace("_", " ").title(),
                   color=_ERROR_COLORS.get(err_type, "#333333"), edgecolor="white", linewidth=0.5)

            for i_bar, val in enumerate(vals):
                if val > 8:
                    txt = ax.text(x[i_bar], bottoms[i_bar] + val / 2, f"{val:.0f}%",
                                  ha="center", va="center", color="white",
                                  fontsize=11, fontweight="bold")
                    txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground="black")])

            bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in plot_models], rotation=30, ha="right", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(f"{mode.capitalize()} Architecture", fontsize=14, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("Percentage of Test Cases (%)", fontsize=14)

    axes[-1].legend(title="Error Taxonomy", bbox_to_anchor=(1.05, 1),
                    loc="upper left", fontsize=12, title_fontsize=13)

    fig.suptitle("Error Breakdown by Architecture and Model",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "error_taxonomy_breakdown")


def _generate_difficulty_breakdown(df: pd.DataFrame, out_dir: str) -> None:
    """Accuracy breakdown by computed difficulty level (L1/L2/L3)."""
    if "TC_Difficulty" not in df.columns:
        return

    modes  = sorted(df["Benchmark_Mode"].unique())
    levels = [lvl for lvl in ["L1", "L2", "L3"] if lvl in df["TC_Difficulty"].unique()]

    if not levels:
        return

    fig, axes = plt.subplots(
        1, len(modes),
        figsize=(max(4, len(levels) * 1.5) * len(modes), 3.5),
        sharey=True,
    )
    if len(modes) == 1:
        axes = [axes]

    palette = {"L1": "#2CA02C", "L2": "#FF7F0E", "L3": "#D62728"}

    for ax, mode in zip(axes, modes):
        mdf = df[df["Benchmark_Mode"] == mode]
        if mdf.empty:
            continue

        grp  = mdf.groupby("TC_Difficulty")["Task_Acc_Validated"].mean() * 100
        x    = np.arange(len(levels))
        vals = [grp.get(lvl, 0) for lvl in levels]

        bars = ax.bar(x, vals, 0.6, color=[palette.get(l, "#333") for l in levels],
                      edgecolor="white", linewidth=1.0, alpha=0.85)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

        ax.set_xticks(x)
        counts = mdf["TC_Difficulty"].value_counts()
        ax.set_xticklabels([f"{lvl}\n(n={counts.get(lvl, 0)})" for lvl in levels], fontsize=11)
        ax.set_ylim(0, 115)
        ax.axhline(100, color="#cccccc", ls="--", lw=0.5)
        ax.set_title(f"{mode.capitalize()} Architecture", fontsize=12, fontweight="bold")

        if ax is axes[0]:
            ax.set_ylabel("Task Accuracy (%)", fontsize=12)

    fig.suptitle("Accuracy by Computed Task Complexity",
                 fontsize=14, fontweight="bold", y=1.05)
    fig.tight_layout()
    _savefig(fig, out_dir, "difficulty_breakdown")


def _generate_token_efficiency_plot(df: pd.DataFrame, out_dir: str) -> None:
    """Scatter plot showing accuracy vs prompt tokens used."""
    if "Prompt_Tokens" not in df.columns or df["Prompt_Tokens"].sum() == 0:
        return

    agg = df.groupby(["Model", "Benchmark_Mode"]).agg(
        avg_tokens=("Prompt_Tokens", "mean"),
        avg_acc=("Task_Acc_Validated", "mean")
    ).reset_index()

    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5))

    modes   = sorted(agg["Benchmark_Mode"].unique())
    markers = {"single": "o", "dual": "s"}

    for mode in modes:
        mdf = agg[agg["Benchmark_Mode"] == mode]
        if mdf.empty:
            continue

        ax.scatter(mdf["avg_tokens"], mdf["avg_acc"] * 100,
                   label=f"{mode.capitalize()} Mode",
                   color=_MODE_PALETTE.get(mode, "#555555"),
                   marker=markers.get(mode, "D"),
                   s=100, alpha=0.8, edgecolor="white", linewidth=1.0, zorder=3)

        for _, row in mdf.iterrows():
            model_name = _short(row["Model"])
            dx, dy = row["avg_tokens"] * 0.02, 0.0
            ha, va = "left", "center"

            # Custom offsets to prevent overlap
            if mode == "dual":
                if model_name == "Gemma-3 1B":
                    dx, dy = -row["avg_tokens"] * 0.025, -1.8
                    ha, va = "right", "top"
                elif model_name == "LFM2.5 Instruct 1.2B":
                    dx, dy = row["avg_tokens"] * 0.025, 1.2
                    ha, va = "left", "bottom"
                elif model_name == "Qwen3.5 0.8B":
                    dx, dy = -row["avg_tokens"] * 0.025, -1.8
                    ha, va = "right", "top"
                elif model_name == "Qwen3.5 2B":
                    dx, dy = row["avg_tokens"] * 0.025, 1.2
                    ha, va = "left", "bottom"

            ax.text(row["avg_tokens"] + dx, row["avg_acc"] * 100 + dy,
                    model_name, fontsize=10.5, fontweight="bold", alpha=0.9, zorder=5,
                    ha=ha, va=va)

    ax.set_xlabel("Average Prompt Tokens per Task", fontsize=12, labelpad=12)
    ax.set_ylabel("Task Accuracy (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title("Token Efficiency: Accuracy vs Context Size", fontweight="bold", fontsize=13)
    ax.set_ylim(0, 105)
    x_max = agg["avg_tokens"].max()
    ax.set_xlim(left=0, right=x_max * 1.2)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _savefig(fig, out_dir, "token_efficiency_scatter")


def _generate_comparison_plots(df: pd.DataFrame, out_dir: str) -> None:
    """Side-by-side comparison of different architectures for paper figures.

    FIX-6: Removed unused `import matplotlib.patches as mpatches`.
    """
    modes   = sorted(df["Benchmark_Mode"].unique())
    n_modes = len(modes)

    grp  = (
        df.groupby(["Benchmark_Mode", "TC_Category"])["Composite_Acc"]
        .mean()
        .reset_index()
    )
    cats = sorted(grp["TC_Category"].unique())
    x    = np.arange(len(cats))

    total_width = 0.8
    w       = total_width / n_modes
    offsets = np.linspace(-total_width / 2 + w / 2, total_width / 2 - w / 2, n_modes)

    fig, ax = plt.subplots(figsize=(max(5, len(cats) * 0.9 + 2), 4))
    for i, mode in enumerate(modes):
        mgrp = grp[grp["Benchmark_Mode"] == mode]
        vals = []
        for c in cats:
            subset = mgrp[mgrp["TC_Category"] == c]["Composite_Acc"]
            vals.append(float(subset.mean() * 100) if not subset.empty else 0.0)

        bars = ax.bar(x + offsets[i], vals, w, label=mode.capitalize(),
                      color=_MODE_PALETTE.get(mode, _PALETTE[i % len(_PALETTE)]),
                      alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("Task Accuracy (%)")
    ax.set_title("Architecture Comparison — Accuracy by Task Category")
    ax.set_ylim(0, 120)
    ax.axhline(100, color="#cccccc", ls="--", lw=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    _savefig(fig, out_dir, "combined_accuracy_by_category")

    # Latency comparison
    all_models = sort_models(df["Model"].unique())
    xi = np.arange(len(all_models))

    fig, ax = plt.subplots(figsize=(max(4, len(modes) * 1.5), 3.5))
    for i, mode in enumerate(modes):
        mdf = df[df["Benchmark_Mode"] == mode]
        ys, err = [], []
        for model in all_models:
            sub = mdf[mdf["Model"] == model]["Total_Time_s"]
            if not sub.empty:
                ys.append(sub.mean())
                err.append(sub.std() if len(sub) > 1 else 0.0)
            else:
                ys.append(0.0)
                err.append(0.0)

        ax.bar(xi + offsets[i], ys, w, label=mode.capitalize(),
               color=_MODE_PALETTE.get(mode, _PALETTE[i % len(_PALETTE)]),
               yerr=err, capsize=3, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(xi)
    ax.set_xticklabels([_short(m) for m in all_models], rotation=20, ha="right")
    ax.set_ylabel("Avg Latency (s)")
    ax.set_title("Architecture Comparison — Latency")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    _savefig(fig, out_dir, "combined_latency_comparison")


def _generate_combined_bfcl_radar(df: pd.DataFrame, out_dir: str) -> None:
    """Radar footprint comparing Single vs Dual (averaged over models)."""
    modes = sorted(df["Benchmark_Mode"].unique())
    if len(modes) < 2:
        return

    pivot = df.pivot_table(
        index="Benchmark_Mode", columns="TC_Category",
        values="Composite_Acc", aggfunc="mean",
    )

    categories = [str(c).replace("_", " ").title() for c in pivot.columns]
    N = len(categories)
    if N < 3:
        return

    angles  = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, size=11, weight="bold")
    ax.tick_params(axis="x", pad=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#343434", size=11)
    plt.ylim(0, 1.05)

    for mode in modes:
        color = _MODE_PALETTE.get(mode, _PALETTE[0])
        vals  = pivot.loc[mode].values.flatten().tolist()
        vals += [vals[0]]
        ax.plot(angles, vals, linewidth=2.5, linestyle="solid",
                label=f"{mode.capitalize()} Architecture", color=color, alpha=0.9)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
               title="Architectures", frameon=True, fontsize=10.5, title_fontsize=11)
    ax.set_title("Combined Task Footprint Comparison\n(Mean across all models)",
              size=13, weight="bold", y=1.1)

    _savefig(fig, out_dir, "combined_bfcl_radar")


def _generate_model_radar_grid(df: pd.DataFrame, out_dir: str) -> None:
    """A grid of radar plots, one per model, comparing Single vs Dual."""
    models = sort_models(df["Model"].unique())
    modes  = sorted(df["Benchmark_Mode"].unique())
    if len(modes) < 2:
        return

    full_pivot = df.pivot_table(
        index=["Model", "Benchmark_Mode"], columns="TC_Category",
        values="Composite_Acc", aggfunc="mean",
    )
    categories = [str(c).replace("_", " ").title() for c in full_pivot.columns]
    N = len(categories)
    if N < 3:
        return

    angles  = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.5, rows * 4.5))

    for i, model in enumerate(models):
        ax = fig.add_subplot(rows, cols, i + 1, polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, size=9, weight="bold")
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [], color="grey", size=8)
        plt.ylim(0, 1.05)

        for mode in modes:
            if (model, mode) in full_pivot.index:
                vals  = full_pivot.loc[(model, mode)].values.flatten().tolist()
                vals += [vals[0]]
                color = _MODE_PALETTE.get(mode, "#555555")
                ax.plot(angles, vals, linewidth=1.8, label=mode.capitalize(),
                        color=color, alpha=0.8)
                ax.fill(angles, vals, color=color, alpha=0.1)

        ax.set_title(_short(model), size=12, weight="bold", pad=20)
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.15, 1.15), fontsize=10)

    fig.suptitle("Per-Model Task Footprint: Architecture Comparison",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir, "combined_model_radar_grid")


def _generate_model_accuracy_by_category(df: pd.DataFrame, out_dir: str) -> None:
    """Accuracy by task category, one subplot per mode, bars grouped by model."""
    modes       = sorted(df["Benchmark_Mode"].unique())
    plot_models = sort_models(df["Model"].unique())

    if not modes:
        print("  [skip] No modes found for model accuracy by category plot.")
        return

    n_models    = len(plot_models)
    cats        = sorted(df["TC_Category"].unique())
    x           = np.arange(len(cats))
    total_width = 0.85
    w           = total_width / n_models
    offsets     = np.linspace(-total_width / 2 + w / 2, total_width / 2 - w / 2, n_models)

    n_modes = len(modes)
    fig, axes = plt.subplots(
        1, n_modes,
        figsize=(max(6, len(cats) * 1.1 + 2) * n_modes, 4.5),
        sharey=True,
    )
    if n_modes == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        for i, model in enumerate(plot_models):
            mdf = df[(df["Model"] == model) & (df["Benchmark_Mode"] == mode)]

            grp  = mdf.groupby("TC_Category")["Composite_Acc"].mean() * 100
            vals = [grp.get(c, 0.0) for c in cats]

            color = _PALETTE[i % len(_PALETTE)]
            alpha = 0.85

            label = _short(model)
            bars  = ax.bar(x + offsets[i], vals, w, label=label,
                           color=color, alpha=alpha, edgecolor="white", linewidth=0.5)

            for bar, v in zip(bars, vals):
                if v > 5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=6, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=30, ha="right")
        ax.set_ylabel("Task Accuracy (%)")
        ax.set_title(f"{mode.capitalize()} Architecture", fontweight="bold")
        ax.set_ylim(0, 115)
        ax.axhline(100, color="#cccccc", ls="--", lw=0.6)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Models", fontsize=7)

    fig.suptitle(
        "Model Comparison — Accuracy by Task Category\n"
        "(Each subplot = one architecture)",
        fontsize=11, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    _savefig(fig, out_dir, "model_accuracy_by_category")


def _generate_model_bfcl_radar_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Radar footprint comparing all models."""
    modes  = sorted(df["Benchmark_Mode"].unique())
    models = sort_models(df["Model"].unique())

    if not modes or len(models) < 1:
        print("  [skip] Not enough models/modes for model BFCL radar comparison.")
        return

    # Build one radar per mode
    for mode in modes:
        mode_df = df[df["Benchmark_Mode"] == mode]

        pivot = mode_df.pivot_table(
            index="Model", columns="TC_Category",
            values="Composite_Acc", aggfunc="mean",
        ).reindex(models)

        categories = [str(c).replace("_", " ").title() for c in pivot.columns]
        N = len(categories)
        if N < 3:
            continue

        angles  = [n / N * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, size=8, weight="bold")
        ax.tick_params(axis="x", pad=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
                   ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
        plt.ylim(0, 1.05)

        # Model series
        for i, model in enumerate(models):
            if model not in pivot.index:
                continue
            color = _PALETTE[i % len(_PALETTE)]
            vals  = pivot.loc[model].values.flatten().tolist()
            vals += [vals[0]]
            ax.plot(angles, vals, linewidth=2.0, linestyle="solid",
                    label=_short(model), color=color, alpha=0.85)
            ax.fill(angles, vals, color=color, alpha=0.05)

        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
                   title="Models", frameon=True, fontsize=8)
        ax.set_title(
            f"Model Footprint — {mode.capitalize()} Architecture",
            size=11, weight="bold", y=1.1,
        )

        _savefig(fig, out_dir, f"model_bfcl_radar_{mode}")


def _generate_model_difficulty_breakdown_comparison(df: pd.DataFrame, out_dir: str) -> None:
    """Accuracy by difficulty level, grouped by model."""
    if "TC_Difficulty" not in df.columns:
        return

    models = sort_models(df["Model"].unique())
    levels = [lvl for lvl in ["L1", "L2", "L3"] if lvl in df["TC_Difficulty"].unique()]

    if not levels or not models:
        return

    n_models = len(models)
    x        = np.arange(n_models)
    w        = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n_models * 1.3 + 2), 6.5))

    palette = {"L1": "#2CA02C", "L2": "#FF7F0E", "L3": "#D62728"}

    for i, lvl in enumerate(levels):
        offset = (i - (len(levels) - 1) / 2) * w
        vals = []
        for model in models:
            mdf = df[(df["Model"] == model) & (df["TC_Difficulty"] == lvl)]
            vals.append(mdf["Task_Acc_Validated"].mean() * 100 if not mdf.empty else 0)

        bars = ax.bar(x + offset, vals, w, label=lvl, color=palette.get(lvl, "#333"),
                      alpha=0.8, edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_short(m) for m in models], rotation=25, ha="right", fontsize=16)
    ax.set_ylabel("Task Accuracy (%)", fontsize=17)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_title("Model Comparison — Accuracy by Task Complexity",
                 fontweight="bold", fontsize=18)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="#cccccc", ls="--", lw=0.6)
    ax.legend(title="Complexity", loc="upper left", bbox_to_anchor=(1, 1), fontsize=15, title_fontsize=16)

    fig.tight_layout()
    _savefig(fig, out_dir, "model_difficulty_breakdown")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from expense_benchmark.py output."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to combined (or single/dual) Excel or CSV file from expense_benchmark.py",
    )
    parser.add_argument(
        "--output", default="figures/",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--mode", default=None,
        choices=["single", "dual"],
        help="Only plot this mode. Default: plot all modes found in the file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found.")
        return

    os.makedirs(args.output, exist_ok=True)
    df = load_and_clean_data(args.input)
    generate_plots(df, args.output, mode_filter=args.mode)
    print(f"\nDone. Plots saved to: {args.output}")


if __name__ == "__main__":
    main()