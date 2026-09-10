# utils/llm_input_validation.py
import pandas as pd
import difflib
from typing import Dict, Any, Tuple, Optional

def get_metadata_lists(df: pd.DataFrame):
    """
    Extract category and major_category lists from the dataframe.
    Deduplicate case-variant keys.
    """
    if "category" not in df.columns:
        return {}, {}

    categories = set(df["category"].dropna().unique())
    major_categories = set()
    if "major category" in df.columns:
        major_categories = set(df["major category"].dropna().unique())

    category_lookup: dict[str, str] = {}
    for cat in categories:
        key = str(cat).lower()
        existing = category_lookup.get(key)
        if existing is None or str(cat) == key:
            category_lookup[key] = str(cat)

    major_category_lookup: dict[str, str] = {}
    for cat in major_categories:
        key = str(cat).lower()
        existing = major_category_lookup.get(key)
        if existing is None or str(cat) == key:
            major_category_lookup[key] = str(cat)

    return category_lookup, major_category_lookup

def validate_and_fix_params(
    params: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Validates and auto-corrects the category / major_category fields in LLM output.
    """
    category_lookup, major_category_lookup = get_metadata_lists(df)
    cleaned_params = params.copy()
    warnings: list[str] = []

    # Remap year/month to start_year/start_month if end_year/end_month is provided
    if 'end_year' in cleaned_params or 'end_month' in cleaned_params:
        if 'year' in cleaned_params and 'start_year' not in cleaned_params:
            cleaned_params['start_year'] = cleaned_params.pop('year')
            warnings.append("Mapped 'year' to 'start_year' for date range")
        if 'month' in cleaned_params and 'start_month' not in cleaned_params:
            cleaned_params['start_month'] = cleaned_params.pop('month')
            warnings.append("Mapped 'month' to 'start_month' for date range")

    input_cat = params.get("category") or params.get("major_category")

    if isinstance(input_cat, str):
        input_cat = input_cat.strip("`").strip("'").strip('"').strip()

    if not input_cat:
        return cleaned_params, None

    def _token_overlap_score(value: str, candidate: str) -> float:
        v_toks = set(value.lower().split())
        c_toks = set(candidate.lower().split())
        if not v_toks or not c_toks:
            return 0.0
        return len(v_toks & c_toks) / len(v_toks | c_toks)

    def find_best_match(value: str, lookup_dict: dict[str, str]):
        if not value:
            return None, 0.0
        val_lower = str(value).lower()

        if val_lower in lookup_dict:
            return lookup_dict[val_lower], 1.0

        if val_lower.endswith("s") and val_lower[:-1] in lookup_dict:
            return lookup_dict[val_lower[:-1]], 0.9
        if val_lower.endswith("ies") and (val_lower[:-3] + "y") in lookup_dict:
            return lookup_dict[val_lower[:-3] + "y"], 0.9

        matches = difflib.get_close_matches(val_lower, lookup_dict.keys(), n=1, cutoff=0.80)
        if matches:
            return lookup_dict[matches[0]], 0.8

        best_key, best_score = None, 0.0
        for candidate_key in lookup_dict:
            score = _token_overlap_score(val_lower, candidate_key)
            if score > best_score:
                best_score, best_key = score, candidate_key
        if best_score >= 0.50:
            return lookup_dict[best_key], round(best_score * 0.7, 3)

        return None, 0.0

    match_major, score_major = find_best_match(input_cat, major_category_lookup)
    match_cat, score_cat = find_best_match(input_cat, category_lookup)

    if score_major >= 0.8 and score_major >= score_cat:
        cleaned_params.pop("major_category", None)
        cleaned_params["category"] = match_major
        if input_cat.lower() != str(match_major).lower():
            warnings.append(f"Mapped '{input_cat}' → major_category='{match_major}'")

    elif score_cat >= 0.8:
        cleaned_params.pop("major_category", None)
        cleaned_params["category"] = match_cat
        if input_cat.lower() != str(match_cat).lower():
            warnings.append(f"Corrected '{input_cat}' → category='{match_cat}'")

    elif score_cat >= 0.50:
        cleaned_params.pop("major_category", None)
        cleaned_params["category"] = match_cat
        warnings.append(
            f"Low-confidence match: '{input_cat}' → category='{match_cat}' "
            f"(score={score_cat:.2f}). Verify manually."
        )

    else:
        warnings.append(
            f"No category match for '{input_cat}' (best score={max(score_major, score_cat):.2f}). "
            f"Original value retained."
        )

    warning_msg = " | ".join(warnings) if warnings else None
    return cleaned_params, warning_msg