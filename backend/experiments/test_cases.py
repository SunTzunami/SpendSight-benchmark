# experiments/test_cases.py
"""
Test suite for the SpendSight tool-calling benchmark.

Each test case has a *computed* complexity score based on 7 measurable features:
  - n_params:  number of expected parameters
  - d_date:    date specification complexity (0=none, 1=year, 2=year+month, 3=range, 4=day)
  - d_cat:     category normalization distance (0=exact, 1=plural/minor, 2=semantic mapping)
  - d_rel:     uses relative time expression (0/1)
  - d_multi:   compound/multi-value param groups (e.g. comparison with 2 date sets)
  - d_abbr:    contains abbreviated years/informal date formats (0/1)
  - d_holiday: requires semantic calendar/holiday knowledge (0/2)

complexity = n_params + d_date + d_cat + d_rel + d_multi + d_abbr + d_holiday
Binning: L1 (<=4), L2 (5-7), L3 (>=8)
"""

from typing import Any
import re


# ── Complexity scoring ──────────────────────────────────────────────────────

def _compute_date_complexity(expected: dict[str, Any]) -> int:
    """Score the date specification complexity of a test case."""
    has_day = "day" in expected or "d1" in expected or "d2" in expected
    has_range = ("start_year" in expected and "end_year" in expected) or \
                ("y1" in expected and "y2" in expected)
    has_month = "month" in expected or "start_month" in expected or \
                "end_month" in expected or "m1" in expected or "m2" in expected or \
                "sm1" in expected or "em1" in expected or "sm2" in expected or "em2" in expected
    has_year = "year" in expected or "start_year" in expected or \
               "y1" in expected or "y2" in expected
    has_relative = "months" in expected

    if has_day:
        return 4
    if has_range and has_month:
        return 3
    if has_range:
        return 3
    if has_relative:
        return 2
    if has_year and has_month:
        return 2
    if has_year:
        return 1
    return 0


def _compute_category_distance(query: str, expected: dict[str, Any]) -> int:
    """Score how far the query's natural phrasing is from the expected category name.

    Returns 2 for any unmapped category (e.g. "supplements", "snacks", "gym", "clothing", "books").
    This is intentional to represent that mapped concepts require a genuine semantic leap/domain knowledge 
    to match from informal queries.
    """
    cat = expected.get("category") or expected.get("major_category")
    if cat is None:
        return 0

    q_lower = query.lower()
    cat_lower = str(cat).lower()

    # Exact match in query
    if cat_lower in q_lower:
        return 0

    # Semantic mappings that require domain knowledge.
    # Key = canonical category name (lowercase), value = list of natural-language aliases.
    # Distance-1 aliases (trivial mapping): plural, abbreviation, one-word shortening.
    semantic_maps = {
        # Food
        "dining":               ["eating out", "eat out", "dine"],
        "combini meal":         ["combini", "combinis", "convenience store"],
        "café":                 ["cafe", "cafes"],
        # Housing and Utilities
        "gas bill":             ["gas"],
        "electricity bill":     ["electricity"],
        "water & sewage bill":  ["water bill"],
        "internet bill":        ["internet", "wifi"],
        "phone bill":           ["phone", "mobile bill"],
        # Transportation
        "flight tickets":       ["flights", "flight"],
        "tokyo metro":          ["metro", "subway"],
        "ride share":           ["rideshare"],
        "shinkansen":           ["bullet train"],
        # Fitness
        "futsal game":          ["futsal"],
        "basketball game":      ["basketball"],
        "football game":        ["football"],
        "sports event":         ["sports events", "sporting events", "sporting event"],
        # Entertainment
        "arcades & karaoke":    ["karaoke", "karaokes", "arcade", "arcades"],
        "events & venues":      ["events", "venue", "venues"],
        "nomikai":              ["nomikais", "work drinks", "work party"],
        # Souvenirs
        "souvenirs/gifts/treats": ["souvenirs and treats", "souvenirs", "gifts"],
        # Household
        "housing and utilities":  ["utilities"],
        # Misc
        "healthcare":           ["medical", "doctor", "hospital"],
        "personal care":        ["hygiene", "toiletries"],
        # New mappings for v2 test cases
        "grocery":              ["groceries", "grocery bills"],
        "medicines":            ["meds", "medication"],
        "bus":                  ["bus fares", "bus fare"],
        "car rental":           ["car rentals"],
        "donation":             ["donations"],
        "taxi":                 ["taxis"],
    }

    for canonical, aliases in semantic_maps.items():
        if cat_lower == canonical:
            if any(alias in q_lower for alias in aliases):
                return 1  # Minor mapping (common alias)
                
    # Everything else → distance 2
    return 2  # Requires real semantic mapping


def _has_relative_time(query: str) -> int:
    """Check if query uses relative time expressions."""
    return 1 if re.search(r"\b(past|last)\s+\d*\s*(month|year|months|years)\b", query.lower()) else 0


def _has_abbreviations(query: str) -> int:
    """Check if query uses abbreviated year forms or informal date notation."""
    q = query.lower()
    # Require apostrophe specifically for the year-abbreviation case: '24, '25, '26 etc.
    if re.search(r"'\d{2}\b", q):
        return 1
    # Slash notation: 2024/01, 11/2025
    if re.search(r"\d{4}/\d{1,2}|\d{1,2}/\d{4}", q):
        return 1
    return 0


def _count_multi_value_groups(expected: dict[str, Any]) -> int:
    """Count compound parameter groups (e.g., comparison needs 2 date sets)."""
    count = 0
    if "y1" in expected and "y2" in expected:
        count += 1
    if ("m1" in expected and "m2" in expected) or ("sm1" in expected and "sm2" in expected):
        count += 1
    return count


def _has_holiday_knowledge(query: str) -> int:
    """Check if query requires knowledge of specific holidays/informal calendar names."""
    q = query.lower()
    holiday_terms = [
        "new year's eve", "new years eve", "nye",
        "new year's day", "new years day",
        "christmas eve", "christmas", "xmas"
    ]
    if any(term in q for term in holiday_terms):
        return 2
    return 0


def compute_complexity(tc: dict[str, Any]) -> tuple[int, str]:
    """Compute complexity score and level for a test case.

    Returns: (score, level) where level is 'L1', 'L2', or 'L3'.
    """
    expected = tc["expected"]
    query = tc["q"]

    n_params = len(expected)
    d_date = _compute_date_complexity(expected)
    d_cat = _compute_category_distance(query, expected)
    d_rel = _has_relative_time(query)
    d_multi = _count_multi_value_groups(expected)
    d_abbr = _has_abbreviations(query)
    d_holiday = _has_holiday_knowledge(query)

    score = n_params + d_date + d_cat + d_rel + d_multi + d_abbr + d_holiday

    if score <= 4:
        level = "L1"
    elif score <= 7:
        level = "L2"
    else:
        level = "L3"

    return score, level



# ── Test cases ──────────────────────────────────────────────────────────────

_RAW_CASES: list[dict[str, Any]] = [
    # ── Time series (10) ────────────────────────────────────────────────────
    {
        "id": "TS01", "group": "time_series",
        "q": "Can ya show spending on food for past 6 months?",
        "tool": "plot_time_series",
        "expected": {"category": "Food", "months": 6},
    },
    {
        "id": "TS02", "group": "time_series",
        "q": "plot spend at cafes from june 24 to feb 2026",
        "tool": "plot_time_series",
        "expected": {"category": "cafe", "start_year": 2024, "start_month": 6, "end_year": 2026, "end_month": 2},
    },
    {
        "id": "TS03", "group": "time_series",
        "q": "plot spend on snacks for 2024 oct to 2025 dec",
        "tool": "plot_time_series",
        "expected": {"category": "snacks", "start_year": 2024, "start_month": 10, "end_year": 2025, "end_month": 12},
    },
    {
        "id": "TS04", "group": "time_series",
        "q": "show spend at eating out from 2025 june to april 2026",
        "tool": "plot_time_series",
        "expected": {"category": "dining", "start_year": 2025, "start_month": 6, "end_year": 2026, "end_month": 4},
    },
    {
        "id": "TS05", "group": "time_series",
        "q": "plot spend on fitness for past month",
        "tool": "plot_time_series",
        "expected": {"category": "Fitness", "months": 1},
    },
    {
        "id": "TS06", "group": "time_series",
        "q": "plot spend on supplements for past year",
        "tool": "plot_time_series",
        "expected": {"category": "supplements", "months": 12},
    },
    {
        "id": "TS07", "group": "time_series",
        "q": "show spend at combinis for 2024 oct to 2025 dec",
        "tool": "plot_time_series",
        "expected": {"category": "combini meal", "start_year": 2024, "start_month": 10, "end_year": 2025, "end_month": 12},
    },
    {
        "id": "TS08", "group": "time_series",
        "q": "make a plot to show spend on nomikais from 2024/01 to 2026/04",
        "tool": "plot_time_series",
        "expected": {"category": "nomikai", "start_year": 2024, "start_month": 1, "end_year": 2026, "end_month": 4},
    },
    {
        "id": "TS09", "group": "time_series",
        "q": "make a plot to show spend on souvenirs and treats for past year.",
        "tool": "plot_time_series",
        # FIX: lowercased to match sub-category casing convention used throughout suite.
        # validate_and_fix_params maps this up to the major category "Souvenirs/Gifts/Treats".
        "expected": {"category": "souvenirs/gifts/treats", "months": 12},
    },
    {
        "id": "TS10", "group": "time_series",
        "q": "plot spending on gas for past 30 months",
        "tool": "plot_time_series",
        "expected": {"category": "gas bill", "months": 30},
    },

    # ── Time series (11-20) ──────────────────────────────────────────────────
    {
        "id": "TS11", "group": "time_series",
        "q": "show me how my groceries spending has changed over 2025",
        "tool": "plot_time_series",
        "expected": {"category": "grocery", "year": 2025},
    },
    {
        "id": "TS12", "group": "time_series",
        "q": "plot trend for bullet train expenses from jan 2024 to dec 2025",
        "tool": "plot_time_series",
        "expected": {"category": "shinkansen", "start_year": 2024, "start_month": 1, "end_year": 2025, "end_month": 12},
    },
    {
        "id": "TS13", "group": "time_series",
        "q": "can u show my accommodation spending for past 3 months?",
        "tool": "plot_time_series",
        "expected": {"category": "Accommodation", "months": 3},
    },
    {
        "id": "TS14", "group": "time_series",
        "q": "show electronics spending since 2023",
        "tool": "plot_time_series",
        "expected": {"category": "electronics", "start_year": 2023},
    },
    {
        "id": "TS15", "group": "time_series",
        "q": "plot spend on tuition from 2024/04 to 2026/03",
        "tool": "plot_time_series",
        "expected": {"category": "tuition", "start_year": 2024, "start_month": 4, "end_year": 2026, "end_month": 3},
    },
    {
        "id": "TS16", "group": "time_series",
        "q": "show me my transportation spending from sep 2024 to march 2025",
        "tool": "plot_time_series",
        "expected": {"category": "Transportation", "start_year": 2024, "start_month": 9, "end_year": 2025, "end_month": 3},
    },
    {
        "id": "TS17", "group": "time_series",
        "q": "plot spending on meds for past 18 months",
        "tool": "plot_time_series",
        "expected": {"category": "medicines", "months": 18},
    },
    {
        "id": "TS18", "group": "time_series",
        "q": "show my ride share costs from '24 to 26",
        "tool": "plot_time_series",
        "expected": {"category": "ride share", "start_year": 2024, "end_year": 2026},
    },
    {
        "id": "TS19", "group": "time_series",
        "q": "plot spending from 6/2023 to dec '25",
        "tool": "plot_time_series",
        "expected": {"start_year": 2023, "start_month": 6, "end_year": 2025, "end_month": 12},
    },
    {
        "id": "TS20", "group": "time_series",
        "q": "show spend on basketball from oct 24 to mar 26",
        "tool": "plot_time_series",
        "expected": {"category": "basketball game", "start_year": 2024, "start_month": 10, "end_year": 2026, "end_month": 3},
    },
    {
        "id": "TS21", "group": "time_series",
        "q": "Show spending trend for past 4 months exclude rent tho",
        "tool": "plot_time_series",
        "expected": {"months": 4, "ignore_rent": True},
    },
    {
        "id": "TS22", "group": "time_series",
        "q": "Plot my monthly expenses over the past year excluding rent",
        "tool": "plot_time_series",
        "expected": {"months": 12, "ignore_rent": True},
    },
    {
        "id": "TS23", "group": "time_series",
        "q": "plot spend on cafes for past 1.5 yrs",
        "tool": "plot_time_series",
        "expected": {"category": "cafe", "months": 18},
    },

    # ── Distribution (10) ───────────────────────────────────────────────────
    {
        "id": "DI01", "group": "distribution",
        "q": "show breakdown of education related expenses for 2025...",
        "tool": "plot_distribution",
        "expected": {"category": "Education", "year": 2025},
    },
    {
        "id": "DI02", "group": "distribution",
        "q": "plz share distribuution of food expenses for 2025...",
        "tool": "plot_distribution",
        "expected": {"category": "Food", "year": 2025},
    },
    {
        "id": "DI03", "group": "distribution",
        # FIX: query was "2025/06 to 2025/02" (inverted). Corrected to match expected Feb→Jun.
        "q": "plz share breakdown on expenses related to education for 2025/02 to 2025/06.",
        "tool": "plot_distribution",
        "expected": {"category": "Education", "start_year": 2025, "start_month": 2, "end_year": 2025, "end_month": 6},
    },
    {
        "id": "DI04", "group": "distribution",
        "q": "show breakdown of spend on utilities for past year",
        "tool": "plot_distribution",
        "expected": {"category": "Housing and Utilities", "months": 12},
    },
    {
        "id": "DI05", "group": "distribution",
        "q": "show breakdown of spend on entertainment for 2024",
        "tool": "plot_distribution",
        "expected": {"category": "Entertainment", "year": 2024},
    },
    {
        "id": "DI06", "group": "distribution",
        "q": "show breakdown of expenses for june 21 2024",
        "tool": "plot_distribution",
        "expected": {"year": 2024, "month": 6, "day": 21},
    },
    {
        "id": "DI07", "group": "distribution",
        "q": "show breakdown of expenses for 2025 april",
        "tool": "plot_distribution",
        "expected": {"year": 2025, "month": 4},
    },
    {
        "id": "DI08", "group": "distribution",
        "q": "show breakdown of fitness expenses for dec 2024",
        "tool": "plot_distribution",
        "expected": {"category": "Fitness", "year": 2024, "month": 12},
    },
    {
        "id": "DI09", "group": "distribution",
        "q": "tell me breakdown of spend on transportation for past 2 years",
        "tool": "plot_distribution",
        "expected": {"category": "Transportation", "months": 24},
    },
    {
        "id": "DI10", "group": "distribution",
        "q": "show breakdown of food expenses for past 3 months",
        "tool": "plot_distribution",
        "expected": {"category": "Food", "months": 3},
    },

    # ── Distribution (11-20) ─────────────────────────────────────────────────
    {
        "id": "DI11", "group": "distribution",
        "q": "how is my spending split for 2025 excluding rent?",
        "tool": "plot_distribution",
        "expected": {"year": 2025, "ignore_rent": True},
    },
    {
        "id": "DI12", "group": "distribution",
        "q": "gimme a pie chart of accomodation expenses from 2024 - 25",
        "tool": "plot_distribution",
        "expected": {"category": "Accommodation", "start_year": 2024, "end_year": 2025},
    },
    {
        "id": "DI13", "group": "distribution",
        "q": "show me breakdown of miscellaneous expenses for past 6 months",
        "tool": "plot_distribution",
        "expected": {"category": "Miscellaneous", "months": 6},
    },
    {
        "id": "DI14", "group": "distribution",
        "q": "share distro of household, clothing expenses.....",
        "tool": "plot_distribution",
        "expected": {"category": "Household and Clothing"},
    },
    {
        "id": "DI15", "group": "distribution",
        "q": "what does my electronics and furniture spending look like for jan 2025?",
        "tool": "plot_distribution",
        "expected": {"category": "Electronics and Furniture", "year": 2025, "month": 1},
    },
    {
        "id": "DI16", "group": "distribution",
        "q": "breakdown of all expenses for 15 march 2025",
        "tool": "plot_distribution",
        "expected": {"year": 2025, "month": 3, "day": 15},
    },
    {
        "id": "DI17", "group": "distribution",
        "q": "can ya show breakdown of spending w/o rent for past 3 months?",
        "tool": "plot_distribution",
        "expected": {"months": 3, "ignore_rent": True},
    },
    {
        "id": "DI18", "group": "distribution",
        "q": "plz share distribuution of souvenirs expenses from 2023 jan to 2025/12",
        "tool": "plot_distribution",
        "expected": {"category": "souvenirs/gifts/treats", "start_year": 2023, "start_month": 1, "end_year": 2025, "end_month": 12},
    },
    {
        "id": "DI19", "group": "distribution",
        "q": "show breakdown of entertainment expenses for nov 2024 to feb 2025",
        "tool": "plot_distribution",
        "expected": {"category": "Entertainment", "start_year": 2024, "start_month": 11, "end_year": 2025, "end_month": 2},
    },
    {
        "id": "DI20", "group": "distribution",
        "q": "how did I split my money in 2024?",
        "tool": "plot_distribution",
        "expected": {"year": 2024},
    },
    {
        "id": "DI21", "group": "distribution",
        "q": "show breakdown of all expenses for mar 2026 (exclude rent tho)",
        "tool": "plot_distribution",
        "expected": {"year": 2026, "month": 3, "ignore_rent": True},
    },

    # ── Comparison bars (10) ────────────────────────────────────────────────
    {
        "id": "CP01", "group": "comparison",
        "q": "compare spend on electricity 2025 vs 26.",
        "tool": "plot_comparison_bars",
        "expected": {"category": "electricity bill", "y1": 2025, "y2": 2026},
    },
    {
        "id": "CP02", "group": "comparison",
        "q": "contrast spending on groceries jan 2025 vs jan 2026",
        "tool": "plot_comparison_bars",
        "expected": {"category": "grocery", "y1": 2025, "m1": 1, "y2": 2026, "m2": 1},
    },
    {
        "id": "CP03", "group": "comparison",
        "q": "contrast spending at nomikais 2024 vs 25..",
        "tool": "plot_comparison_bars",
        "expected": {"category": "nomikai", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP04", "group": "comparison",
        "q": "compare spend on gym in nov 2024 vs nov 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "gym", "y1": 2024, "m1": 11, "y2": 2025, "m2": 11},
    },
    {
        "id": "CP05", "group": "comparison",
        "q": "compare spend on electricity 24 vs 25",
        "tool": "plot_comparison_bars",
        "expected": {"category": "electricity bill", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP06", "group": "comparison",
        "q": "compare spend on food for 31 dec 2024 vs same date in 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "Food", "y1": 2024, "m1": 12, "d1": 31, "y2": 2025, "m2": 12, "d2": 31},
    },
    {
        "id": "CP07", "group": "comparison",
        "q": "compare spend on flights 2024 vs 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "flight tickets", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP08", "group": "comparison",
        "q": "compare spending on snacks for jan 2025 vs july 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "snacks", "y1": 2025, "m1": 1, "y2": 2025, "m2": 7},
    },
    {
        "id": "CP09", "group": "comparison",
        "q": "can ya compare spend on water bill for 2024 dec vs 2025 dec?",
        "tool": "plot_comparison_bars",
        "expected": {"category": "water & sewage bill", "y1": 2024, "m1": 12, "y2": 2025, "m2": 12},
    },
    {
        "id": "CP10", "group": "comparison",
        "q": "compare spend on eating out 2025 vs 2026..",
        "tool": "plot_comparison_bars",
        "expected": {"category": "dining", "y1": 2025, "y2": 2026},
    },

    # ── Comparison bars (11-20) ──────────────────────────────────────────────
    {
        "id": "CP11", "group": "comparison",
        "q": "compare grocery spending feb 2025 vs same month in 26",
        "tool": "plot_comparison_bars",
        "expected": {"category": "grocery", "y1": 2025, "m1": 2, "y2": 2026, "m2": 2},
    },
    {
        "id": "CP12", "group": "comparison",
        "q": "how does my accommodation spend in 2024 compare to 25?",
        "tool": "plot_comparison_bars",
        "expected": {"category": "Accommodation", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP13", "group": "comparison",
        "q": "contrast transportation costs for mar 2025 vs march 2026",
        "tool": "plot_comparison_bars",
        "expected": {"category": "Transportation", "y1": 2025, "m1": 3, "y2": 2026, "m2": 3},
    },
    {
        "id": "CP14", "group": "comparison",
        "q": "compare spend on wifi 2024 vs 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "internet bill", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP15", "group": "comparison",
        "q": "compare spend on phone 2025 vs 26",
        "tool": "plot_comparison_bars",
        "expected": {"category": "phone bill", "y1": 2025, "y2": 2026},
    },
    {
        "id": "CP16", "group": "comparison",
        "q": "compare spend on clothing for jul 2024 vs jul 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "clothing", "y1": 2024, "m1": 7, "y2": 2025, "m2": 7},
    },
    {
        "id": "CP17", "group": "comparison",
        "q": "compare education expenses '24 vs '25",
        "tool": "plot_comparison_bars",
        "expected": {"category": "Education", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP18", "group": "comparison",
        "q": "compare overall spending for 10 jan 2025 vs same date in 2026",
        "tool": "plot_comparison_bars",
        "expected": {"y1": 2025, "m1": 1, "d1": 10, "y2": 2026, "m2": 1, "d2": 10},
    },
    {
        "id": "CP19", "group": "comparison",
        "q": "difference in subway spending between 2024 and 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "tokyo metro", "y1": 2024, "y2": 2025},
    },
    {
        "id": "CP20", "group": "comparison",
        "q": "compare household expenses for april '24 vs apr 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "household", "y1": 2024, "m1": 4, "y2": 2025, "m2": 4},
    },
    {
        "id": "CP21", "group": "comparison",
        "q": "compare overall spend in '24 vs '25 (exclude rent tho)",
        "tool": "plot_comparison_bars",
        "expected": {"y1": 2024, "y2": 2025, "ignore_rent": True},
    },
    {
        "id": "CP22", "group": "comparison",
        "q": "compare total spend on new years eve 2024 vs for same day on 2025",
        "tool": "plot_comparison_bars",
        "expected": {"y1": 2024, "m1": 12, "d1": 31, "y2": 2025, "m2": 12, "d2": 31},
    },
    {
        "id": "CP23", "group": "comparison",
        "q": "compare dining between Jan-Jun 2024 and Jan-Jun 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "dining", "y1": 2024, "sm1": 1, "em1": 6, "y2": 2025, "sm2": 1, "em2": 6},
    },
    {
        "id": "CP24", "group": "comparison",
        "q": "compare groceries nov 2024 to april 2025 vs nov 2025 to april 2026",
        "tool": "plot_comparison_bars",
        "expected": {"category": "grocery", "y1": 2024, "sm1": 11, "ey1": 2025, "em1": 4, "y2": 2025, "sm2": 11, "ey2": 2026, "em2": 4},
    },
    {
        "id": "CP25", "group": "comparison",
        "q": "compare overall expenses from oct 2024 to dec 2024 vs oct 2025 to dec 2025 excluding rent",
        "tool": "plot_comparison_bars",
        "expected": {"y1": 2024, "sm1": 10, "em1": 12, "y2": 2025, "sm2": 10, "em2": 12, "ignore_rent": True},
    },
    {
        "id": "CP26", "group": "comparison",
        "q": "compare electricity from jan-mar '24 vs jan-mar '25",
        "tool": "plot_comparison_bars",
        "expected": {"category": "electricity bill", "y1": 2024, "sm1": 1, "em1": 3, "y2": 2025, "sm2": 1, "em2": 3},
    },
    {
        "id": "CP27", "group": "comparison",
        "q": "compare groceries jan 2024 - april 2024 vs jan 25 - apr 2025",
        "tool": "plot_comparison_bars",
        "expected": {"category": "grocery", "y1": 2024, "sm1": 1, "em1": 4, "y2": 2025, "sm2": 1, "em2": 4},
    },
    {
        "id": "CP28", "group": "comparison",
        "q": "compare groceries 11/2024 - 04/2025 vs 11/2025-04/2026",
        "tool": "plot_comparison_bars",
        "expected": {"category": "grocery", "y1": 2024, "sm1": 11, "ey1": 2025, "em1": 4, "y2": 2025, "sm2": 11, "ey2": 2026, "em2": 4},
    },

    # ── Calculate total (10) ────────────────────────────────────────────────
    {
        "id": "CT01", "group": "calculate_total",
        "q": "can ya tell total spend on combini food for 2025?",
        "tool": "calculate_total",
        "expected": {"category": "combini meal", "year": 2025},
    },
    {
        "id": "CT02", "group": "calculate_total",
        "q": "tell me total spent on water bill for 2025",
        "tool": "calculate_total",
        "expected": {"category": "water & sewage bill", "year": 2025},
    },
    {
        "id": "CT03", "group": "calculate_total",
        "q": "tell sum spent at karaoke from 2023 to 2026",
        "tool": "calculate_total",
        "expected": {"category": "arcades & karaoke", "start_year": 2023, "end_year": 2026},
    },
    {
        "id": "CT04", "group": "calculate_total",
        "q": "tell me total spent on souvenirs from 2024 to 2025..",
        "tool": "calculate_total",
        "expected": {"category": "souvenirs/gifts/treats", "start_year": 2024, "end_year": 2025},
    },
    {
        "id": "CT05", "group": "calculate_total",
        "q": "tell me sum spent on treats from 2023 to 2026..",
        "tool": "calculate_total",
        "expected": {"category": "souvenirs/gifts/treats", "start_year": 2023, "end_year": 2026},
    },
    {
        "id": "CT06", "group": "calculate_total",
        "q": "tell me sum spent on gas for 2024",
        "tool": "calculate_total",
        "expected": {"category": "gas bill", "year": 2024},
    },
    {
        "id": "CT07", "group": "calculate_total",
        "q": "can ya sum total spent on metro for 2024 july?",
        "tool": "calculate_total",
        # FIX: was "Tokyo Metro" (title case). Lowercased to match all other sub-categories.
        "expected": {"category": "tokyo metro", "year": 2024, "month": 7},
    },
    {
        "id": "CT08", "group": "calculate_total",
        "q": "Can ya tell total spend on food for past month?",
        "tool": "calculate_total",
        "expected": {"category": "Food", "months": 1},
    },
    {
        "id": "CT09", "group": "calculate_total",
        "q": "can ya sum total spend on snacks for past 6 months?",
        "tool": "calculate_total",
        "expected": {"category": "snacks", "months": 6},
    },
    {
        "id": "CT10", "group": "calculate_total",
        "q": "can ya tell me total spend at combinis in 2025?",
        "tool": "calculate_total",
        "expected": {"category": "combini meal", "year": 2025},
    },

    # ── Calculate total (11-20) ──────────────────────────────────────────────
    {
        "id": "CT11", "group": "calculate_total",
        "q": "how much did I spend on groceries in 2025?",
        "tool": "calculate_total",
        "expected": {"category": "grocery", "year": 2025},
    },
    {
        "id": "CT12", "group": "calculate_total",
        "q": "total spent on bus fares from 2024 to 2025",
        "tool": "calculate_total",
        "expected": {"category": "bus", "start_year": 2024, "end_year": 2025},
    },
    {
        "id": "CT13", "group": "calculate_total",
        "q": "how much did I spend on car rentals for past 2 years?",
        "tool": "calculate_total",
        "expected": {"category": "car rental", "months": 24},
    },
    {
        "id": "CT14", "group": "calculate_total",
        "q": "total amount spent on exam fees in 2025",
        "tool": "calculate_total",
        "expected": {"category": "exam fees", "year": 2025},
    },
    {
        "id": "CT15", "group": "calculate_total",
        "q": "whats the total spend on donations from '23 to 2026?",
        "tool": "calculate_total",
        "expected": {"category": "donation", "start_year": 2023, "end_year": 2026},
    },
    {
        "id": "CT16", "group": "calculate_total",
        "q": "how much on taxis in dec 2024?",
        "tool": "calculate_total",
        "expected": {"category": "taxi", "year": 2024, "month": 12},
    },
    {
        "id": "CT17", "group": "calculate_total",
        "q": "total spent on personal care for past 3 months",
        "tool": "calculate_total",
        "expected": {"category": "personal care", "months": 3},
    },
    {
        "id": "CT18", "group": "calculate_total",
        "q": "how much did I spend at cable car from 2024 june to march 2025?",
        "tool": "calculate_total",
        "expected": {"category": "cable car", "start_year": 2024, "start_month": 6, "end_year": 2025, "end_month": 3},
    },
    {
        "id": "CT19", "group": "calculate_total",
        "q": "sum of all spending on 22 feb 2025",
        "tool": "calculate_total",
        "expected": {"year": 2025, "month": 2, "day": 22},
    },
    {
        "id": "CT20", "group": "calculate_total",
        "q": "what was the total spent on books in 2024?",
        "tool": "calculate_total",
        "expected": {"category": "books", "year": 2024},
    },
    {
        "id": "CT21", "group": "calculate_total",
        "q": "How much did I spend in the last 6 months? (no rent)",
        "tool": "calculate_total",
        "expected": {"months": 6, "ignore_rent": True},
    },
    {
        "id": "CT22", "group": "calculate_total",
        "q": "Excluding rent, what was my total spend in 2025?",
        "tool": "calculate_total",
        "expected": {"year": 2025, "ignore_rent": True},
    },

    # ── Top expenses (10) ───────────────────────────────────────────────────
    {
        "id": "TP01", "group": "top_expenses",
        "q": "can ya get me top 5 expenses for 31 dec 2025?",
        "tool": "get_top_expenses",
        "expected": {"n": 5, "year": 2025, "month": 12, "day": 31},
    },
    {
        "id": "TP02", "group": "top_expenses",
        "q": "tell me top 8 expenses for 11/2025..",
        "tool": "get_top_expenses",
        "expected": {"n": 8, "year": 2025, "month": 11},
    },
    {
        "id": "TP03", "group": "top_expenses",
        "q": "can ya tell me the top 3 expenses for 2024?",
        "tool": "get_top_expenses",
        "expected": {"n": 3, "year": 2024},
    },
    {
        "id": "TP04", "group": "top_expenses",
        "q": "can ya get me top 10 eating out expenses in 2025?",
        "tool": "get_top_expenses",
        "expected": {"n": 10, "category": "dining", "year": 2025},
    },
    {
        "id": "TP05", "group": "top_expenses",
        "q": "tell me the top 4 expenses from 2023 to 2026.",
        "tool": "get_top_expenses",
        "expected": {"n": 4, "start_year": 2023, "end_year": 2026},
    },
    {
        "id": "TP06", "group": "top_expenses",
        "q": "what were the top 5 expenses for past month? disregard rent",
        "tool": "get_top_expenses",
        "expected": {"n": 5, "months": 1, "ignore_rent": True},
    },
    {
        "id": "TP07", "group": "top_expenses",
        "q": "tell me top 10 expenses of 2025 nov w/o rent",
        "tool": "get_top_expenses",
        "expected": {"n": 10, "year": 2025, "month": 11, "ignore_rent": True},
    },
    {
        "id": "TP08", "group": "top_expenses",
        "q": "can ya tell me top 7 futsal expenses for 2025?",
        "tool": "get_top_expenses",
        "expected": {"n": 7, "category": "futsal game", "year": 2025},
    },
    {
        "id": "TP09", "group": "top_expenses",
        "q": "can ya tell me about the top 3 expenses on clothing in 2024?",
        "tool": "get_top_expenses",
        "expected": {"n": 3, "category": "clothing", "year": 2024},
    },
    {
        "id": "TP10", "group": "top_expenses",
        "q": "tell me about top 10 expenses of last year but exclude rent tho",
        "tool": "get_top_expenses",
        "expected": {"n": 10, "months": 12, "ignore_rent": True},
    },

    # ── Top expenses (11-20) ────────────────────────────────────────────────
    {
        "id": "TP11", "group": "top_expenses",
        "q": "top 5 transportation expenses in 2025",
        "tool": "get_top_expenses",
        "expected": {"n": 5, "category": "Transportation", "year": 2025},
    },
    {
        "id": "TP12", "group": "top_expenses",
        "q": "what are the 3 biggest grocery bills in 2024?",
        "tool": "get_top_expenses",
        "expected": {"n": 3, "category": "grocery", "year": 2024},
    },
    {
        "id": "TP13", "group": "top_expenses",
        "q": "show me top 10 expenses from jan 2025 to june '25",
        "tool": "get_top_expenses",
        "expected": {"n": 10, "start_year": 2025, "start_month": 1, "end_year": 2025, "end_month": 6},
    },
    {
        "id": "TP14", "group": "top_expenses",
        "q": "top 6 entertainment expenses for past 6 months",
        "tool": "get_top_expenses",
        "expected": {"n": 6, "category": "Entertainment", "months": 6},
    },
    {
        "id": "TP15", "group": "top_expenses",
        "q": "biggest 5 expenses on 1 jan 2025",
        "tool": "get_top_expenses",
        "expected": {"n": 5, "year": 2025, "month": 1, "day": 1},
    },
    {
        "id": "TP16", "group": "top_expenses",
        "q": "top 10 expenses for past 3 months exclude rent",
        "tool": "get_top_expenses",
        "expected": {"n": 10, "months": 3, "ignore_rent": True},
    },
    {
        "id": "TP17", "group": "top_expenses",
        "q": "show top 8 accommodation expenses from 2024 to 2025",
        "tool": "get_top_expenses",
        "expected": {"n": 8, "category": "Accommodation", "start_year": 2024, "end_year": 2025},
    },
    {
        "id": "TP18", "group": "top_expenses",
        "q": "most expensive 4 purchases on education in 2025",
        "tool": "get_top_expenses",
        "expected": {"n": 4, "category": "Education", "year": 2025},
    },
    {
        "id": "TP19", "group": "top_expenses",
        "q": "tell me the top 15 expenses for 2025, ignore rent plz",
        "tool": "get_top_expenses",
        "expected": {"n": 15, "year": 2025, "ignore_rent": True},
    },
    {
        "id": "TP20", "group": "top_expenses",
        "q": "what were the top 5 sporting events expenses for past year?",
        "tool": "get_top_expenses",
        "expected": {"n": 5, "category": "sports event", "months": 12},
    },
    {
        "id": "TP21", "group": "top_expenses",
        "q": "tell me about top 10 expenses from feb 2024 to jan 2026 (exclude rent tho)",
        "tool": "get_top_expenses",
        "expected": {"n": 10, "start_year": 2024, "start_month": 2, "end_year": 2026, "end_month": 1, "ignore_rent": True},
    }
]

# ── Compute complexity and build final TEST_CASES ───────────────────────────

TEST_CASES: list[dict[str, Any]] = []
for tc in _RAW_CASES:
    score, level = compute_complexity(tc)
    TEST_CASES.append({
        **tc,
        "complexity_score": score,
        "difficulty": level,
    })