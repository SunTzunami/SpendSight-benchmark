#!/usr/bin/env python3
"""
Embedding-similarity baseline for tool routing.

Compares cosine-similarity-based intent classification (using a small
sentence-transformer) against the LLM router on the 115-question
SpendSight benchmark.

Usage:
    python backend/experiments/embedding_baseline.py
"""

import sys, os, json

# Prevent transformers from attempting to import broken TF installation
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import Counter

# ── Setup path so we can import test_cases ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from test_cases import TEST_CASES


# ── Tool descriptions (mirrors router_prompt.txt) ─────────────────────────
# Each description is the natural-language description the router sees.
TOOL_DESCRIPTIONS = {
    "plot_time_series": (
        "Plot a time series chart showing spending trends over time. "
        "Use for: trends, spending over time, past X months, since a year, date ranges. "
        "Keywords: trend, over time, months, years, since, plot, show."
    ),
    "plot_distribution": (
        "Plot a distribution breakdown or pie chart of expenses by category. "
        "Use for: breakdown, distribution, proportions, pie chart, how is spending split. "
        "Keywords: breakdown, distribution, pie chart, split, exclude rent, without rent."
    ),
    "plot_comparison_bars": (
        "Plot comparison bar charts comparing spending between two different time periods. "
        "Use for: comparing two periods such as Dec 2024 vs Dec 2025. "
        "Keywords: compare, vs, versus, difference between, contrast."
    ),
    "calculate_total": (
        "Calculate the total amount spent. "
        "Use for: simple totals, sums, specific amounts, how much did I spend. "
        "Keywords: how much, total, sum, cost, amount."
    ),
    "get_top_expenses": (
        "Get the top N biggest or most expensive items. "
        "Use for: biggest expenses, largest expenses, top X items, most expensive. "
        "Keywords: biggest, largest, top, most expensive, highest."
    ),
}

TOOL_NAMES = list(TOOL_DESCRIPTIONS.keys())
TOOL_TEXTS = list(TOOL_DESCRIPTIONS.values())


# ── Mean-pooling helper ────────────────────────────────────────────────────
def mean_pooling(model_output, attention_mask):
    """Mean pool token embeddings, respecting attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode(texts: list[str], tokenizer, model) -> np.ndarray:
    """Encode a list of texts into L2-normalised embeddings."""
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    embs = mean_pooling(outputs, encoded["attention_mask"])
    embs = F.normalize(embs, p=2, dim=1)
    return embs.numpy()


def run_baseline(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print(f"\n{'='*70}")
    print(f"  Embedding-Similarity Baseline: {model_name}")
    print(f"{'='*70}\n")

    # 1. Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    print("  Model loaded.\n")

    # 2. Encode tool descriptions
    tool_embs = encode(TOOL_TEXTS, tokenizer, model)  # (5, dim)

    # 3. Encode all queries
    queries = [tc["q"] for tc in TEST_CASES]
    query_embs = encode(queries, tokenizer, model)     # (115, dim)

    # 4. Cosine similarity → predicted tool
    # sim[i, j] = cosine similarity between query i and tool j
    sim = query_embs @ tool_embs.T  # (115, 5)

    predictions = [TOOL_NAMES[int(np.argmax(sim[i]))] for i in range(len(queries))]
    ground_truth = [tc["tool"] for tc in TEST_CASES]

    # 5. Compute metrics
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    total = len(ground_truth)
    overall_acc = correct / total * 100

    # Per-tool breakdown
    tool_counts: dict[str, dict[str, int]] = {}
    for tool in TOOL_NAMES:
        tool_counts[tool] = {"correct": 0, "total": 0}

    for pred, gt in zip(predictions, ground_truth):
        tool_counts[gt]["total"] += 1
        if pred == gt:
            tool_counts[gt]["correct"] += 1

    # Per-difficulty breakdown
    diff_counts: dict[str, dict[str, int]] = {}
    for tc, pred in zip(TEST_CASES, predictions):
        d = tc["difficulty"]
        if d not in diff_counts:
            diff_counts[d] = {"correct": 0, "total": 0}
        diff_counts[d]["total"] += 1
        if pred == tc["tool"]:
            diff_counts[d]["correct"] += 1

    # 6. Print results
    print(f"Overall Tool-Selection Accuracy: {correct}/{total} = {overall_acc:.1f}%\n")

    print("Per-Tool Breakdown:")
    print(f"  {'Tool':<25} {'Correct':>7} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10}")
    for tool in TOOL_NAMES:
        c = tool_counts[tool]["correct"]
        t = tool_counts[tool]["total"]
        acc = c / t * 100 if t > 0 else 0
        print(f"  {tool:<25} {c:>7} {t:>7} {acc:>9.1f}%")

    print(f"\nPer-Difficulty Breakdown:")
    print(f"  {'Level':<8} {'Correct':>7} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*10}")
    for level in sorted(diff_counts.keys()):
        c = diff_counts[level]["correct"]
        t = diff_counts[level]["total"]
        acc = c / t * 100 if t > 0 else 0
        print(f"  {level:<8} {c:>7} {t:>7} {acc:>9.1f}%")

    # 7. Confusion matrix (compact)
    print(f"\nConfusion Matrix (rows=ground truth, cols=predicted):")
    print(f"  {'':>25}", end="")
    for t in TOOL_NAMES:
        print(f"  {t[:8]:>8}", end="")
    print()
    for gt_tool in TOOL_NAMES:
        print(f"  {gt_tool:>25}", end="")
        for pred_tool in TOOL_NAMES:
            count = sum(1 for g, p in zip(ground_truth, predictions) if g == gt_tool and p == pred_tool)
            print(f"  {count:>8}", end="")
        print()

    # 8. Show misclassified examples (first 10)
    errors = [(tc, pred) for tc, pred in zip(TEST_CASES, predictions) if tc["tool"] != pred]
    if errors:
        print(f"\nMisclassified Examples ({len(errors)} total, showing first 15):")
        for tc, pred in errors[:15]:
            print(f"  [{tc['id']}] \"{tc['q'][:60]}...\"")
            print(f"    GT: {tc['tool']}  →  Predicted: {pred}")
    else:
        print("\nNo misclassifications!")

    print(f"\n{'='*70}")
    print(f"  Summary: Embedding baseline achieves {overall_acc:.1f}% tool-selection accuracy")
    print(f"  vs. LLM router (reference: check per-model results in paper)")
    print(f"{'='*70}\n")

    return overall_acc, tool_counts, diff_counts


if __name__ == "__main__":
    run_baseline()
