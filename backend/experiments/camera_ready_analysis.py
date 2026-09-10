#!/usr/bin/env python3
"""
Camera-ready re-analysis script for SpendSight (FinNLP 2026).

Reads existing benchmark CSV (8,050 observations) and produces:
  1. Bootstrap confidence intervals (supplementary table)
  2. Single-agent per-group and per-tier breakdowns
  3. Complexity feature frequency statistics
  4. Few-shot / test-set overlap analysis
  5. Single-agent ACS values for Table 4

All outputs saved to: benchmark_outputs/camera_ready_outputs/ (override with --output).

Usage:
    python experiments/camera_ready_analysis.py \\
        --input benchmark_outputs/run_5reps_combined.csv \\
        --output benchmark_outputs/camera_ready_outputs/
"""

import argparse
import os
import sys
import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher

# ── paths (defaults, overridable via CLI) ──────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(BACKEND_DIR, "benchmark_outputs", "run_5reps_combined.csv")
OUTPUT_DIR = os.path.join(BACKEND_DIR, "benchmark_outputs", "camera_ready_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add experiments dir to path so we can import test_cases
sys.path.insert(0, os.path.join(BACKEND_DIR, "experiments"))
sys.path.insert(0, SCRIPT_DIR)
from test_cases import (
    _RAW_CASES, compute_complexity,
    _compute_date_complexity, _compute_category_distance,
    _has_relative_time, _count_multi_value_groups,
    _has_abbreviations, _has_holiday_knowledge
)

# Pretty model names
MODEL_NAMES = {
    "EXAONE-4.0-1.2B-Q8_0": "EXAONE-4.0-1.2B",
    "gemma-3-1b-it-Q8_0": "Gemma-3-1B-it",
    "LFM2.5-1.2B-Instruct-Q8_0": "LFM2.5-1.2B",
    "minicpm5-1b-Q8_0": "MiniCPM5-1B",
    "Qwen3.5-0.8B-Q8_0": "Qwen3.5-0.8B",
    "Qwen3.5-2B-Q8_0": "Qwen3.5-2B",
    "ai9stars_G9v3-3B-Q4_K_M": "G9v3-3B",
}

MODEL_ORDER = [
    "EXAONE-4.0-1.2B", "Gemma-3-1B-it", "LFM2.5-1.2B",
    "MiniCPM5-1B", "Qwen3.5-0.8B", "Qwen3.5-2B", "G9v3-3B"
]

GROUP_NAMES = {
    "time_series": "Time Series",
    "distribution": "Distribution",
    "comparison": "Comparison",
    "calculate_total": "Calc. Total",
    "top_expenses": "Top Expenses",
}


def load_data():
    """Load and preprocess the benchmark CSV."""
    df = pd.read_csv(CSV_PATH)
    df["Model_Short"] = df["Model"].map(MODEL_NAMES)
    df["TC_Group_Pretty"] = df["TC_Group"].map(GROUP_NAMES)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

def compute_bootstrap_cis(df, n_resamples=2000, seed=42):
    """
    Compute 95% bootstrap CIs for Task Accuracy and CR per model x mode.
    Resampling unit: test cases (115 items), each averaged over 5 reps.
    """
    rng = np.random.RandomState(seed)
    results = []

    for mode in ["single", "dual"]:
        mode_df = df[df["Benchmark_Mode"] == mode]

        for model_short in MODEL_ORDER:
            model_df = mode_df[mode_df["Model_Short"] == model_short]
            if model_df.empty:
                continue

            # Average over 5 reps per test case
            tc_means = model_df.groupby("TC_ID").agg({
                "Task_Acc_Validated": "mean",
                "Correct_Ratio_Validated": "mean",
                "FSP_Validated": "mean",
            }).reset_index()

            n_tc = len(tc_means)
            ta_values = tc_means["Task_Acc_Validated"].values * 100
            cr_values = tc_means["Correct_Ratio_Validated"].values * 100
            fsp_values = tc_means["FSP_Validated"].values * 100

            # Bootstrap
            ta_boots = []
            cr_boots = []
            fsp_boots = []
            for _ in range(n_resamples):
                idx = rng.choice(n_tc, size=n_tc, replace=True)
                ta_boots.append(np.mean(ta_values[idx]))
                cr_boots.append(np.mean(cr_values[idx]))
                fsp_boots.append(np.mean(fsp_values[idx]))

            ta_boots = np.array(ta_boots)
            cr_boots = np.array(cr_boots)
            fsp_boots = np.array(fsp_boots)

            results.append({
                "Mode": mode.capitalize(),
                "Model": model_short,
                "Task_Acc": f"{np.mean(ta_values):.1f}",
                "Task_Acc_CI": f"[{np.percentile(ta_boots, 2.5):.1f}, {np.percentile(ta_boots, 97.5):.1f}]",
                "CR": f"{np.mean(cr_values):.1f}",
                "CR_CI": f"[{np.percentile(cr_boots, 2.5):.1f}, {np.percentile(cr_boots, 97.5):.1f}]",
                "FSP": f"{np.mean(fsp_values):.1f}",
                "FSP_CI": f"[{np.percentile(fsp_boots, 2.5):.1f}, {np.percentile(fsp_boots, 97.5):.1f}]",
            })

    results_df = pd.DataFrame(results)

    # Generate LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{95\% bootstrap confidence intervals ($n=2000$ resamples over 115 test cases, each averaged across 5 repetitions). Point estimates match Table~\ref{tab:main_results}.}",
        r"\label{tab:bootstrap_cis}",
        r"\begin{tabular}{@{}ll|cc|cc|cc@{}}",
        r"\toprule",
        r"\textbf{Mode} & \textbf{Model} & \textbf{Task Acc} & \textbf{95\% CI} & \textbf{CR} & \textbf{95\% CI} & \textbf{FSP} & \textbf{95\% CI} \\",
        r"\midrule",
    ]

    prev_mode = None
    for _, row in results_df.iterrows():
        if row["Mode"] != prev_mode and prev_mode is not None:
            latex_lines.append(r"\midrule")
        prev_mode = row["Mode"]
        suffix = r"$^\dagger$" if row["Model"] == "G9v3-3B" else ""
        latex_lines.append(
            f"{row['Mode']} & {row['Model']}{suffix} & {row['Task_Acc']} & {row['Task_Acc_CI']} & "
            f"{row['CR']} & {row['CR_CI']} & {row['FSP']} & {row['FSP_CI']} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(latex_lines)

    with open(os.path.join(OUTPUT_DIR, "bootstrap_ci_table.tex"), "w") as f:
        f.write(latex)

    results_df.to_csv(os.path.join(OUTPUT_DIR, "bootstrap_cis.csv"), index=False)
    print("==> Bootstrap CIs saved")
    print(results_df.to_string(index=False))
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. SINGLE-AGENT PER-GROUP AND PER-TIER BREAKDOWNS
# ═══════════════════════════════════════════════════════════════════════════

def compute_single_agent_breakdowns(df):
    """
    Compute single-agent Task Accuracy and CR by tool group and by L1/L2/L3.
    This fills the gap identified by 9C4z Q3-ii.
    """
    single_df = df[df["Benchmark_Mode"] == "single"]

    # --- Per tool group ---
    group_results = []
    for model_short in MODEL_ORDER:
        model_df = single_df[single_df["Model_Short"] == model_short]
        for group_key, group_name in GROUP_NAMES.items():
            grp = model_df[model_df["TC_Group"] == group_key]
            if grp.empty:
                continue
            ta = grp["Task_Acc_Validated"].mean() * 100
            cr = grp["Correct_Ratio_Validated"].mean() * 100
            group_results.append({
                "Model": model_short,
                "Group": group_name,
                "Task_Acc": f"{ta:.1f}",
                "CR": f"{cr:.1f}",
            })

    group_df = pd.DataFrame(group_results)

    # --- Per complexity tier ---
    tier_results = []
    for model_short in MODEL_ORDER:
        model_df = single_df[single_df["Model_Short"] == model_short]
        for tier in ["L1", "L2", "L3"]:
            grp = model_df[model_df["TC_Difficulty"] == tier]
            if grp.empty:
                continue
            ta = grp["Task_Acc_Validated"].mean() * 100
            cr = grp["Correct_Ratio_Validated"].mean() * 100
            tier_results.append({
                "Model": model_short,
                "Tier": tier,
                "Task_Acc": f"{ta:.1f}",
                "CR": f"{cr:.1f}",
            })

    tier_df = pd.DataFrame(tier_results)

    # Generate LaTeX for per-tier single-agent
    tier_latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Single-agent Task Accuracy and CR by complexity tier (validated, \%). Companion to Figure~\ref{fig:difficulty_plot} which shows dual-agent only.}",
        r"\label{tab:single_tier_results}",
        r"\begin{tabular}{@{}l|cc|cc|cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{\textbf{L1}} & \multicolumn{2}{c|}{\textbf{L2}} & \multicolumn{2}{c}{\textbf{L3}} \\",
        r"\textbf{Model} & \textbf{T.A} & \textbf{CR} & \textbf{T.A} & \textbf{CR} & \textbf{T.A} & \textbf{CR} \\",
        r"\midrule",
    ]

    for model_short in MODEL_ORDER:
        suffix = r"$^\dagger$" if model_short == "G9v3-3B" else ""
        row = f"{model_short}{suffix}"
        for tier in ["L1", "L2", "L3"]:
            match = tier_df[(tier_df["Model"] == model_short) & (tier_df["Tier"] == tier)]
            if match.empty:
                row += " & -- & --"
            else:
                row += f" & {match.iloc[0]['Task_Acc']} & {match.iloc[0]['CR']}"
        row += r" \\"
        tier_latex_lines.append(row)

    tier_latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(OUTPUT_DIR, "single_agent_per_tier.tex"), "w") as f:
        f.write("\n".join(tier_latex_lines))

    # Generate per-group table (for top models only, to match dual-agent Table 5 format)
    group_order = ["Time Series", "Distribution", "Comparison", "Calc. Total", "Top Expenses"]
    top_models = ["Qwen3.5-0.8B", "Qwen3.5-2B"]

    grp_latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Single-agent per-group results (validated, \%). Companion to Table~\ref{tab:category_results} which shows dual-agent.}",
        r"\label{tab:single_category_results}",
        r"\begin{tabular}{@{}l|cc|cc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{\textbf{Qwen3.5-0.8B}} & \multicolumn{2}{c}{\textbf{Qwen3.5-2B}} \\",
        r"\textbf{Group} & \textbf{T.Acc} & \textbf{CR} & \textbf{T.Acc} & \textbf{CR} \\",
        r"\midrule",
    ]

    for group_name in group_order:
        row = group_name
        for model_short in top_models:
            match = group_df[(group_df["Model"] == model_short) & (group_df["Group"] == group_name)]
            if match.empty:
                row += " & -- & --"
            else:
                row += f" & {match.iloc[0]['Task_Acc']} & {match.iloc[0]['CR']}"
        row += r" \\"
        grp_latex.append(row)

    grp_latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(OUTPUT_DIR, "single_agent_per_group.tex"), "w") as f:
        f.write("\n".join(grp_latex))

    group_df.to_csv(os.path.join(OUTPUT_DIR, "single_agent_per_group.csv"), index=False)
    tier_df.to_csv(os.path.join(OUTPUT_DIR, "single_agent_per_tier.csv"), index=False)
    print("\n==> Single-agent breakdowns saved")
    print("\nPer-group (top models):")
    print(group_df[group_df["Model"].isin(top_models)].to_string(index=False))
    print("\nPer-tier:")
    print(tier_df.to_string(index=False))
    return group_df, tier_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPLEXITY FEATURE FREQUENCY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_complexity_stats():
    """
    Count how many of the 115 items have nonzero d_hol, d_abbr, d_rel, etc.
    Answers 9C4z Q3-i.
    """
    feature_counts = defaultdict(int)
    category_counts = defaultdict(int)

    feature_details = []

    for tc in _RAW_CASES:
        expected = tc["expected"]
        query = tc["q"]

        n_params = len(expected)
        d_date = _compute_date_complexity(expected)
        d_cat = _compute_category_distance(query, expected)
        d_rel = _has_relative_time(query)
        d_multi = _count_multi_value_groups(expected)
        d_abbr = _has_abbreviations(query)
        d_hol = _has_holiday_knowledge(query)

        score, level = compute_complexity(tc)

        feature_details.append({
            "TC_ID": tc["id"],
            "Group": tc["group"],
            "n_params": n_params,
            "d_date": d_date,
            "d_cat": d_cat,
            "d_rel": d_rel,
            "d_multi": d_multi,
            "d_abbr": d_abbr,
            "d_hol": d_hol,
            "C": score,
            "Level": level,
            "Category": expected.get("category", "(none)"),
        })

        if d_hol > 0: feature_counts["d_hol > 0"] += 1
        if d_abbr > 0: feature_counts["d_abbr > 0"] += 1
        if d_rel > 0: feature_counts["d_rel > 0"] += 1
        if d_multi > 0: feature_counts["d_multi > 0"] += 1
        if d_cat > 0: feature_counts["d_cat > 0"] += 1
        if d_date > 0: feature_counts["d_date > 0"] += 1

        cat = expected.get("category", None)
        if cat:
            category_counts[cat] += 1
        else:
            category_counts["(no category)"] += 1

    # Check for misspellings and colloquialisms
    misspelling_count = 0
    colloquialism_count = 0
    for tc in _RAW_CASES:
        q = tc["q"].lower()
        if any(m in q for m in ["distribuution", "expensees", "comparre", "comparision"]):
            misspelling_count += 1
        if any(c in q for c in ["can ya", "ya ", "tho", "plz", "nah", "tbh"]):
            colloquialism_count += 1

    details_df = pd.DataFrame(feature_details)

    print("\n==> Complexity feature frequencies (out of 115 test cases):")
    for feat, count in sorted(feature_counts.items()):
        print(f"  {feat}: {count} ({count/115*100:.1f}%)")

    print(f"\n  Queries with misspellings: {misspelling_count}")
    print(f"  Queries with colloquialisms: {colloquialism_count}")

    print(f"\n  Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Generate LaTeX
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Complexity feature frequencies across 115 test cases. Each row shows how many test cases have a nonzero value for that complexity feature.}",
        r"\label{tab:complexity_freq}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"\textbf{Feature} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]

    feature_display = {
        "d_date > 0": r"$d_{\text{date}} > 0$ (date specification)",
        "d_cat > 0": r"$d_{\text{cat}} > 0$ (category mapping)",
        "d_rel > 0": r"$d_{\text{rel}} > 0$ (relative time)",
        "d_multi > 0": r"$d_{\text{multi}} > 0$ (compound params)",
        "d_abbr > 0": r"$d_{\text{abbr}} > 0$ (abbreviated dates)",
        "d_hol > 0": r"$d_{\text{hol}} > 0$ (holiday knowledge)",
    }

    for feat_key in ["d_date > 0", "d_cat > 0", "d_rel > 0", "d_multi > 0", "d_abbr > 0", "d_hol > 0"]:
        count = feature_counts.get(feat_key, 0)
        display = feature_display[feat_key]
        latex_lines.append(f"{display} & {count} & {count/115*100:.1f} \\\\")

    latex_lines.append(r"\midrule")
    latex_lines.append(f"Queries with colloquialisms & {colloquialism_count} & {colloquialism_count/115*100:.1f} \\\\")
    latex_lines.append(f"Queries with misspellings & {misspelling_count} & {misspelling_count/115*100:.1f} \\\\")
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(OUTPUT_DIR, "complexity_feature_freq.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    details_df.to_csv(os.path.join(OUTPUT_DIR, "complexity_feature_details.csv"), index=False)
    print("==> Complexity stats saved")
    return details_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEW-SHOT / TEST-SET OVERLAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

FEWSHOT_EXAMPLES = {
    "plot_time_series": [
        {"q": "How much did I spend on futsal for the past 6 months?", "params": {"category": "futsal game", "months": 6}},
        {"q": "Show me food spending from 2023 to 2025", "params": {"category": "Food", "start_year": 2023, "end_year": 2025}},
        {"q": "Plot spending on snacks from oct 2024 to dec 2025", "params": {"category": "snacks", "start_year": 2024, "start_month": 10, "end_year": 2025, "end_month": 12}},
        {"q": "Can ya plot spending on futsal on dec 2024?", "params": {"category": "futsal game", "year": 2024, "month": 12}},
        {"q": "Gym expenses in 2024?", "params": {"category": "gym", "year": 2024}},
        {"q": "Show me my spending trend for the last 3 months", "params": {"months": 3}},
        {"q": "plot spending from june 2023 to dec 2025", "params": {"start_year": 2023, "start_month": 6, "end_year": 2025, "end_month": 12}},
        {"q": "Trend of electricity bills since 2022", "params": {"category": "electricity bill", "start_year": 2022}},
        {"q": "How has my transportation spending changed over time?", "params": {"category": "Transportation"}},
        {"q": "Show spending on combinis for past year", "params": {"category": "combini meal", "months": 12}},
        {"q": "Show spending trend for past 4 months exclude rent tho", "params": {"months": 4, "ignore_rent": True}},
        {"q": "plot spend on cafes for past 2.5 yrs", "params": {"category": "cafe", "months": 30}},
        {"q": "show my ride share costs from '24 to 26", "params": {"category": "ride share", "start_year": 2024, "end_year": 2026}},
        {"q": "show electronics spending since 2023", "params": {"category": "electronics", "start_year": 2023}},
    ],
    "plot_distribution": [
        {"q": "Show me a breakdown of my food expenses in 2024", "params": {"category": "Food", "year": 2024}},
        {"q": "Pie chart of all expenses from Nov 2024 to Feb 2025", "params": {"start_year": 2024, "start_month": 11, "end_year": 2025, "end_month": 2}},
        {"q": "Show me my spending breakdown for last month", "params": {"months": 1}},
        {"q": "Show me my spending breakdown for groceries for past 6 months", "params": {"category": "Grocery", "months": 6}},
        {"q": "show spend distribution for 2026 feb", "params": {"year": 2026, "month": 2}},
        {"q": "show breakdown of all expenses from 2024 to 2025", "params": {"start_year": 2024, "end_year": 2025}},
        {"q": "Show me a breakdown of expenses for Dec 2024 (exclude rent tho)", "params": {"year": 2024, "month": 12, "ignore_rent": True}},
        {"q": "Show spending breakdown for 2023/06/22", "params": {"year": 2023, "month": 6, "day": 22}},
        {"q": "Spending distribution for the last 3 months", "params": {"months": 3}},
        {"q": "Breakdown of transportation spending from 2022 to 2024", "params": {"category": "Transportation", "start_year": 2022, "end_year": 2024}},
        {"q": "How did I split my money between categories in 2023?", "params": {"year": 2023}},
        {"q": "Show me my spending breakdown excluding rent for the past 6 months", "params": {"months": 6, "ignore_rent": True}},
        {"q": "Breakdown of fitness spending for 2025", "params": {"category": "Fitness", "year": 2025}},
        {"q": "Breakdown of my spending since 2024", "params": {"start_year": 2024}},
        {"q": "show breakdown of all expenses for mar 2026 (exclude rent tho)", "params": {"year": 2026, "month": 3, "ignore_rent": True}},
        {"q": "what does my electronics and furniture spending look like for jan 2025?", "params": {"category": "Electronics and Furniture", "year": 2025, "month": 1}},
    ],
    "plot_comparison_bars": [
        {"q": "Compare food spending in 2024 vs 2025", "params": {"category": "Food", "y1": 2024, "y2": 2025}},
        {"q": "Compare dining Jan 2024 vs Jan 2025", "params": {"category": "dining", "y1": 2024, "m1": 1, "y2": 2025, "m2": 1}},
        {"q": "How does my spend on snacks compare between jan 2024 and jan 2026?", "params": {"category": "snacks", "y1": 2024, "m1": 1, "y2": 2026, "m2": 1}},
        {"q": "Compare total spending 2022 vs 2023", "params": {"y1": 2022, "y2": 2023}},
        {"q": "Compare overall expenses from jan 2025 vs jan 2026", "params": {"y1": 2025, "m1": 1, "y2": 2026, "m2": 1}},
        {"q": "Compare transportation on 21 July 2024 vs 21 July 2025", "params": {"category": "Transportation", "y1": 2024, "m1": 7, "d1": 21, "y2": 2025, "m2": 7, "d2": 21}},
        {"q": "Compare spend on electricity 2024 vs 2025", "params": {"category": "electricity bill", "y1": 2024, "y2": 2025}},
        {"q": "Compare gym spending Nov 2024 vs Nov 2025", "params": {"category": "gym", "y1": 2024, "m1": 11, "y2": 2025, "m2": 11}},
        {"q": "Compare my spending in 2024 vs 2025 excluding rent", "params": {"y1": 2024, "y2": 2025, "ignore_rent": True}},
        {"q": "compare education expenses '24 vs '25", "params": {"category": "Education", "y1": 2024, "y2": 2025}},
        {"q": "compare household expenses for april '24 vs apr 2025", "params": {"category": "household", "y1": 2024, "m1": 4, "y2": 2025, "m2": 4}},
        {"q": "Compare dining between Jan-Jun 2024 and Jan-Jun 2025", "params": {"category": "dining", "y1": 2024, "sm1": 1, "em1": 6, "y2": 2025, "sm2": 1, "em2": 6}},
        {"q": "compare electricity from jan-mar '24 vs jan-mar '25", "params": {"category": "electricity bill", "y1": 2024, "sm1": 1, "em1": 3, "y2": 2025, "sm2": 1, "em2": 3}},
        {"q": "Contrast overall spend from Nov 2024 to Feb 2025 vs Nov 2025 to Feb 2026", "params": {"y1": 2024, "sm1": 11, "ey1": 2025, "em1": 2, "y2": 2025, "sm2": 11, "ey2": 2026, "em2": 2}},
        {"q": "compare total spend on new years eve 2024 vs for same day on 2025", "params": {"y1": 2024, "m1": 12, "d1": 31, "y2": 2025, "m2": 12, "d2": 31}},
    ],
    "calculate_total": [
        {"q": "How much did I spend on groceries in Dec 2024?", "params": {"category": "grocery", "year": 2024, "month": 12}},
        {"q": "Total spending from Oct 2024 to March 2025", "params": {"start_year": 2024, "start_month": 10, "end_year": 2025, "end_month": 3}},
        {"q": "What is my total spending in 2025?", "params": {"year": 2025}},
        {"q": "How much did I spend on food in past month?", "params": {"category": "Food", "months": 1}},
        {"q": "Total cost of electricity in 2023", "params": {"category": "electricity bill", "year": 2023}},
        {"q": "Total spent on 15 Jan 2024?", "params": {"year": 2024, "month": 1, "day": 15}},
        {"q": "How much spent on rent in the last 6 months?", "params": {"category": "rent", "months": 6}},
        {"q": "Sum of all transportation expenses in 2025", "params": {"category": "Transportation", "year": 2025}},
        {"q": "Total spent on combini food for 2025?", "params": {"category": "combini meal", "year": 2025}},
        {"q": "How much did I spend in the last 6 months without rent?", "params": {"months": 6, "ignore_rent": True}},
        {"q": "Total spent on dining since 2024", "params": {"category": "dining", "start_year": 2024}},
        {"q": "whats the total spend on donations from '23 to 2026?", "params": {"category": "donation", "start_year": 2023, "end_year": 2026}},
        {"q": "Excluding rent, what was my total spend in 2025?", "params": {"year": 2025, "ignore_rent": True}},
    ],
    "get_top_expenses": [
        {"q": "What were my biggest expenses in Dec 2024?", "params": {"n": 10, "year": 2024, "month": 12}},
        {"q": "Top 5 food expenses in 2024?", "params": {"n": 5, "category": "Food", "year": 2024}},
        {"q": "Top 10 food expenses in june 2025?", "params": {"n": 10, "category": "Food", "year": 2025, "month": 6}},
        {"q": "Top 5 expenses for 26 june 2024?", "params": {"n": 5, "year": 2024, "month": 6, "day": 26}},
        {"q": "Top expenses from July 2023 to Dec 2024", "params": {"start_year": 2023, "start_month": 7, "end_year": 2024, "end_month": 12}},
        {"q": "Show my top 10 expenses from 2023 to 2025", "params": {"n": 10, "start_year": 2023, "end_year": 2025}},
        {"q": "Top 10 expenses of past month?", "params": {"n": 10, "months": 1}},
        {"q": "What are my top expenses excluding rent?", "params": {"n": 10, "ignore_rent": True}},
        {"q": "Show my top 3 largest transactions in 2023", "params": {"n": 3, "year": 2023}},
        {"q": "What were the biggest expenses over the last 3 months, without rent?", "params": {"n": 10, "months": 3, "ignore_rent": True}},
        {"q": "Top 7 futsal expenses for 2025?", "params": {"n": 7, "category": "futsal game", "year": 2025}},
        {"q": "Top 5 expenses last month, exclude rent", "params": {"n": 5, "months": 1, "ignore_rent": True}},
        {"q": "What were my biggest expenses since 2023?", "params": {"n": 10, "start_year": 2023}},
        {"q": "show me top 10 expenses from jan 2025 to june '25", "params": {"n": 10, "start_year": 2025, "start_month": 1, "end_year": 2025, "end_month": 6}},
        {"q": "tell me the top 15 expenses for 2025, ignore rent plz", "params": {"n": 15, "year": 2025, "ignore_rent": True}},
    ],
}


def _normalize_query(q):
    return re.sub(r'[^\w\s]', '', q.lower()).strip()

def _get_param_slots(params):
    return frozenset(params.keys())

def _get_date_pattern(params):
    patterns = set()
    if "months" in params: patterns.add("relative")
    if "year" in params: patterns.add("year")
    if "month" in params: patterns.add("month")
    if "day" in params: patterns.add("day")
    if "start_year" in params or "y1" in params: patterns.add("range")
    if "sm1" in params: patterns.add("multi_range")
    if "ey1" in params: patterns.add("cross_year_range")
    return frozenset(patterns)


def compute_overlap_analysis():
    """Compute structural and lexical overlap between few-shot examples and test cases."""
    all_fewshot = []
    for tool_name, examples in FEWSHOT_EXAMPLES.items():
        for ex in examples:
            all_fewshot.append({
                "q": ex["q"],
                "q_normalized": _normalize_query(ex["q"]),
                "tool": tool_name,
                "param_slots": _get_param_slots(ex["params"]),
                "date_pattern": _get_date_pattern(ex["params"]),
            })

    test_cases = []
    for tc in _RAW_CASES:
        test_cases.append({
            "id": tc["id"],
            "q": tc["q"],
            "q_normalized": _normalize_query(tc["q"]),
            "tool": tc["tool"],
            "param_slots": _get_param_slots(tc["expected"]),
            "date_pattern": _get_date_pattern(tc["expected"]),
        })

    slot_match_count = 0
    date_pattern_match_count = 0
    high_lexical_overlap = 0
    exact_match = 0

    per_tc_results = []

    for tc in test_cases:
        best_lexical = 0
        best_fewshot_q = ""
        slot_matched = False
        date_matched = False

        same_tool_fewshots = [f for f in all_fewshot if f["tool"] == tc["tool"]]

        for fs in same_tool_fewshots:
            sim = SequenceMatcher(None, tc["q_normalized"], fs["q_normalized"]).ratio()
            if sim > best_lexical:
                best_lexical = sim
                best_fewshot_q = fs["q"]
            if tc["param_slots"] == fs["param_slots"]:
                slot_matched = True
            if tc["date_pattern"] == fs["date_pattern"]:
                date_matched = True

        if best_lexical > 0.6: high_lexical_overlap += 1
        if best_lexical >= 0.95: exact_match += 1
        if slot_matched: slot_match_count += 1
        if date_matched: date_pattern_match_count += 1

        per_tc_results.append({
            "TC_ID": tc["id"],
            "Tool": tc["tool"],
            "Best_Lexical_Sim": f"{best_lexical:.3f}",
            "Best_Match_Query": best_fewshot_q,
            "Slot_Match": slot_matched,
            "Date_Pattern_Match": date_matched,
        })

    n = len(test_cases)
    print("\n==> Few-shot / test-set overlap analysis:")
    print(f"  Total test cases: {n}")
    print(f"  Total specialist few-shot examples: {sum(len(v) for v in FEWSHOT_EXAMPLES.values())}")
    print(f"  Exact argument slot match: {slot_match_count}/{n} ({slot_match_count/n*100:.1f}%)")
    print(f"  Date pattern match: {date_pattern_match_count}/{n} ({date_pattern_match_count/n*100:.1f}%)")
    print(f"  High lexical similarity (>0.6): {high_lexical_overlap}/{n} ({high_lexical_overlap/n*100:.1f}%)")
    print(f"  Near-exact match (>0.95): {exact_match}/{n} ({exact_match/n*100:.1f}%)")

    results_df = pd.DataFrame(per_tc_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "fewshot_overlap.csv"), index=False)

    # LaTeX table
    summary_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Structural overlap between specialist few-shot examples and the 115 test cases. Each test case is compared against the few-shot examples for the same tool.}",
        r"\label{tab:fewshot_overlap}",
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"\textbf{Overlap Metric} & \textbf{Count (\%)} \\",
        r"\midrule",
        f"Exact argument slot match & {slot_match_count}/{n} ({slot_match_count/n*100:.1f}\\%) \\\\",
        f"Date pattern match & {date_pattern_match_count}/{n} ({date_pattern_match_count/n*100:.1f}\\%) \\\\",
        f"High lexical similarity ($>$0.6) & {high_lexical_overlap}/{n} ({high_lexical_overlap/n*100:.1f}\\%) \\\\",
        f"Near-exact match ($>$0.95) & {exact_match}/{n} ({exact_match/n*100:.1f}\\%) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(os.path.join(OUTPUT_DIR, "fewshot_overlap.tex"), "w") as f:
        f.write("\n".join(summary_lines))

    # Narrative paragraph
    narrative = (
        f"Of the 115 test cases, {slot_match_count} ({slot_match_count/n*100:.1f}\\%) share the exact same argument "
        f"slot set as at least one specialist few-shot example for the same tool, and "
        f"{date_pattern_match_count} ({date_pattern_match_count/n*100:.1f}\\%) share the same date expression pattern. "
        f"Lexical similarity (SequenceMatcher ratio) between test queries and their nearest specialist few-shot "
        f"query exceeds 0.6 for {high_lexical_overlap} ({high_lexical_overlap/n*100:.1f}\\%) cases, "
        f"with {exact_match} ({exact_match/n*100:.1f}\\%) near-exact matches ($>$0.95). "
        f"This structural similarity is expected since both few-shot examples and test cases exercise "
        f"the same five tools over a fixed parameter schema, but it means that generalization to "
        f"queries with novel slot combinations or phrasings remains to be validated."
    )

    with open(os.path.join(OUTPUT_DIR, "fewshot_overlap_narrative.tex"), "w") as f:
        f.write(narrative)

    print("==> Overlap analysis saved")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
# 5. SINGLE-AGENT ACS VALUES
# ═══════════════════════════════════════════════════════════════════════════

def extract_single_agent_acs(df):
    """Extract single-agent ACS values for Table 4 update."""
    single_df = df[df["Benchmark_Mode"] == "single"]

    print("\n==> Single-agent ACS values (for Table 4):")
    acs_results = []
    for model_short in MODEL_ORDER:
        model_df = single_df[single_df["Model_Short"] == model_short]
        acs = model_df["ACS_Validated"].mean() * 100
        acs_results.append({"Model": model_short, "ACS_Validated": f"{acs:.1f}"})
        print(f"  {model_short}: ACS = {acs:.1f}%")

    acs_df = pd.DataFrame(acs_results)
    acs_df.to_csv(os.path.join(OUTPUT_DIR, "single_agent_acs.csv"), index=False)
    return acs_df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global CSV_PATH, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Camera-ready re-analysis for SpendSight (FinNLP 2026)."
    )
    parser.add_argument(
        "--input", default=CSV_PATH,
        help="Path to combined benchmark CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help="Directory to save outputs (default: %(default)s)",
    )
    args = parser.parse_args()
    CSV_PATH = args.input
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading data from: {CSV_PATH}")
    df = load_data()
    print(f"Loaded {len(df)} observations ({df['TC_ID'].nunique()} test cases x "
          f"{df['Model_Short'].nunique()} models x 2 modes x 5 reps)")

    print("\n" + "=" * 70)
    print("1. BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 70)
    compute_bootstrap_cis(df)

    print("\n" + "=" * 70)
    print("2. SINGLE-AGENT BREAKDOWNS")
    print("=" * 70)
    compute_single_agent_breakdowns(df)

    print("\n" + "=" * 70)
    print("3. COMPLEXITY FEATURE FREQUENCIES")
    print("=" * 70)
    compute_complexity_stats()

    print("\n" + "=" * 70)
    print("4. FEW-SHOT / TEST-SET OVERLAP")
    print("=" * 70)
    compute_overlap_analysis()

    print("\n" + "=" * 70)
    print("5. SINGLE-AGENT ACS VALUES")
    print("=" * 70)
    extract_single_agent_acs(df)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
