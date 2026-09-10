# Tool-specific prompts for Agent 2
# tool_prompts.py

BASE_INSTRUCTIONS = """You are an automated API parameter extractor. Your ONLY purpose is to extract parameters from the user's request and output a SINGLE JSON object containing those parameters.

## CRITICAL RULES (READ CAREFULLY)
1. DO NOT write any Python code or function calls. DO NOT import anything.
2. DO NOT wrap the output in markdown. NO backticks (```).
3. Output EXACTLY and ONLY a JSON object representing the parameters. E.g. {{"category": "Food", "months": 6}}
4. Use EXACT category names from the metadata below. If not an exact match, map it to the closest valid category (e.g. 'groceries' -> 'grocery', 'dining out' -> 'dining').

## Date Rules
- `months=N`: RELATIVE duration (e.g. "past month" -> months=1, "last 3 months" -> months=3, "past year" -> months=12).
- Fractional years: convert to months. "past 0.5 years" -> months=6, "past 2.5 years" -> months=30. 
- Abbreviated years: '24 = 2024, 24 = 2024, ALWAYS output the full 4-digit year.
- `year=YYYY`: specific full calendar year.
- `year=YYYY, month=M`: specific calendar month (e.g. "Jan 2024").
- `year=YYYY, month=M, day=D`: specific calendar date.
- `start_year=YYYY, end_year=YYYY`: specific year range.
- `start_year=YYYY, start_month=M, end_year=YYYY, end_month=M`: specific month-to-month range.
- For comparing two ranges (e.g., comparing Jan-Jun 2024 vs Jan-Jun 2025), use comparison range parameters: `y1=2024, sm1=1, em1=6, y2=2025, sm2=1, em2=6`.
- For comparing cross-year ranges (e.g., Nov 2024-Jan 2025 vs Nov 2025-Jan 2026), use `ey1` and `ey2`: `y1=2024, sm1=11, ey1=2025, em1=1, y2=2025, sm2=11, ey2=2026, em2=1`.
- CRITICAL: For relative queries like "past month", "last month", "last 6 months", ALWAYS use `months=N`. Do NOT use `month=M`.
- Pick ONLY ONE time filter. Do not mix `months` with `year`.

## Available Parameters
{parameters}

## Examples
{examples}

## Context
```
{metadata}
```
Currency: JPY
Today: {current_date}

FINAL REMINDER: Output ONLY the JSON object. NO MARKDOWN, NO BACKTICKS, NO EXPLANATION, NO CODE!
"""

TOOL_PROMPTS = {
    "plot_time_series": {
        "parameters": "category (str, optional), year (int, optional), month (int, optional), start_year (int, optional), start_month (int, optional), end_year (int, optional), end_month (int, optional), months (int, optional), ignore_rent (bool, optional, defaults to false)",
        "examples": """Q: "How much did I spend on futsal for the past 6 months?"
{"category": "futsal game", "months": 6}

Q: "Show me food spending from 2023 to 2025"
{"category": "Food", "start_year": 2023, "end_year": 2025}

Q: "Plot spending on snacks from oct 2024 to dec 2025"
{"category": "snacks", "start_year": 2024, "start_month": 10, "end_year": 2025, "end_month": 12}

Q: "Can ya plot spending on futsal on dec 2024?"
{"category": "futsal game", "year": 2024, "month": 12}

Q: "Gym expenses in 2024?"
{"category": "gym", "year": 2024}

Q: "Show me my spending trend for the last 3 months"
{"months": 3}

Q: "plot spending from june 2023 to dec 2025"
{"start_year": 2023, "start_month": 6, "end_year": 2025, "end_month": 12}

Q: "Trend of electricity bills since 2022"
{"category": "electricity bill", "start_year": 2022}

Q: "How has my transportation spending changed over time?"
{"category": "Transportation"}

Q: "Show spending on combinis for past year"
{"category": "combini meal", "months": 12}

Q: "Show spending trend for past 4 months exclude rent tho"
{"months": 4, "ignore_rent": true}

Q: "plot spend on cafes for past 2.5 yrs"
{"category": "cafe", "months": 30}

Q: "show my ride share costs from '24 to 26"
{"category": "ride share", "start_year": 2024, "end_year": 2026}

Q: "show electronics spending since 2023"
{"category": "electronics", "start_year": 2023}"""
    },

    "plot_distribution": {
        "parameters": "category (str, optional), year (int, optional), month (int, optional), day (int, optional), start_year (int, optional), start_month (int, optional), end_year (int, optional), end_month (int, optional), months (int, optional), ignore_rent (bool, optional, defaults to false)",
        "examples": """Q: "Show me a breakdown of my food expenses in 2024"
{"category": "Food", "year": 2024}

Q: "Pie chart of all expenses from Nov 2024 to Feb 2025"
{"start_year": 2024, "start_month": 11, "end_year": 2025, "end_month": 2}

Q: "Show me my spending breakdown for last month"
{"months": 1}

Q: "Show me my spending breakdown for groceries for past 6 months"
{"category": "Grocery", "months": 6}

Q: "show spend distribution for 2026 feb"
{"year": 2026, "month": 2}

Q: "show breakdown of all expenses from 2024 to 2025"
{"start_year": 2024, "end_year": 2025}

Q: "Show me a breakdown of expenses for Dec 2024 (exclude rent tho)"
{"year": 2024, "month": 12, "ignore_rent": true}

Q: "Show spending breakdown for 2023/06/22"
{"year": 2023, "month": 6, "day": 22}

Q: "Spending distribution for the last 3 months"
{"months": 3}

Q: "Breakdown of transportation spending from 2022 to 2024"
{"category": "Transportation", "start_year": 2022, "end_year": 2024}

Q: "How did I split my money between categories in 2023?"
{"year": 2023}

Q: "Show me my spending breakdown excluding rent for the past 6 months"
{"months": 6, "ignore_rent": true}

Q: "Breakdown of fitness spending for 2025"
{"category": "Fitness", "year": 2025}

Q: "Breakdown of my spending since 2024"
{"start_year": 2024}

Q: "show breakdown of all expenses for mar 2026 (exclude rent tho)"
{"year": 2026, "month": 3, "ignore_rent": true}

Q: "what does my electronics and furniture spending look like for jan 2025?"
{"category": "Electronics and Furniture", "year": 2025, "month": 1}"""
    },

    "plot_comparison_bars": {
        "parameters": "category (str, optional), y1 (int, optional), m1 (int, optional), d1 (int, optional), y2 (int, optional), m2 (int, optional), d2 (int, optional), sm1 (int, optional), em1 (int, optional), sm2 (int, optional), em2 (int, optional), ey1 (int, optional), ey2 (int, optional), ignore_rent (bool, optional, defaults to false)",
        "examples": """Q: "Compare food spending in 2024 vs 2025"
{"category": "Food", "y1": 2024, "y2": 2025}

Q: "Compare dining Jan 2024 vs Jan 2025"
{"category": "dining", "y1": 2024, "m1": 1, "y2": 2025, "m2": 1}

Q: "How does my spend on snacks compare between jan 2024 and jan 2026?"
{"category": "snacks", "y1": 2024, "m1": 1, "y2": 2026, "m2": 1}

Q: "Compare total spending 2022 vs 2023"
{"y1": 2022, "y2": 2023}

Q: "Compare overall expenses from jan 2025 vs jan 2026"
{"y1": 2025, "m1": 1, "y2": 2026, "m2": 1}

Q: "Compare transportation on 21 July 2024 vs 21 July 2025"
{"category": "Transportation", "y1": 2024, "m1": 7, "d1": 21, "y2": 2025, "m2": 7, "d2": 21}

Q: "Compare spend on electricity 2024 vs 2025"
{"category": "electricity bill", "y1": 2024, "y2": 2025}

Q: "Compare gym spending Nov 2024 vs Nov 2025"
{"category": "gym", "y1": 2024, "m1": 11, "y2": 2025, "m2": 11}

Q: "Compare my spending in 2024 vs 2025 excluding rent"
{"y1": 2024, "y2": 2025, "ignore_rent": true}

Q: "compare education expenses '24 vs '25"
{"category": "Education", "y1": 2024, "y2": 2025}

Q: "compare household expenses for april '24 vs apr 2025"
{"category": "household", "y1": 2024, "m1": 4, "y2": 2025, "m2": 4}

Q: "Compare dining between Jan-Jun 2024 and Jan-Jun 2025"
{"category": "dining", "y1": 2024, "sm1": 1, "em1": 6, "y2": 2025, "sm2": 1, "em2": 6}

Q: "compare electricity from jan-mar '24 vs jan-mar '25"
{"category": "electricity bill", "y1": 2024, "sm1": 1, "em1": 3, "y2": 2025, "sm2": 1, "em2": 3}

Q: "Contrast overall spend from Nov 2024 to Feb 2025 vs Nov 2025 to Feb 2026"
{"y1": 2024, "sm1": 11, "ey1": 2025, "em1": 2, "y2": 2025, "sm2": 11, "ey2": 2026, "em2": 2}

Q: "compare total spend on new years eve 2024 vs for same day on 2025"
{"y1": 2024, "m1": 12, "d1": 31, "y2": 2025, "m2": 12, "d2": 31}"""
    },

    "calculate_total": {
        "parameters": "category (str, optional), year (int, optional), month (int, optional), day (int, optional), start_year (int, optional), start_month (int, optional), end_year (int, optional), end_month (int, optional), months (int, optional), ignore_rent (bool, optional, defaults to false)",
        "examples": """Q: "How much did I spend on groceries in Dec 2024?"
{"category": "grocery", "year": 2024, "month": 12}

Q: "Total spending from Oct 2024 to March 2025"
{"start_year": 2024, "start_month": 10, "end_year": 2025, "end_month": 3}

Q: "What is my total spending in 2025?"
{"year": 2025}

Q: "How much did I spend on food in past month?"
{"category": "Food", "months": 1}

Q: "Total cost of electricity in 2023"
{"category": "electricity bill", "year": 2023}

Q: "Total spent on 15 Jan 2024?"
{"year": 2024, "month": 1, "day": 15}

Q: "How much spent on rent in the last 6 months?"
{"category": "rent", "months": 6}

Q: "Sum of all transportation expenses in 2025"
{"category": "Transportation", "year": 2025}

Q: "Total spent on combini food for 2025?"
{"category": "combini meal", "year": 2025}

Q: "How much did I spend in the last 6 months without rent?"
{"months": 6, "ignore_rent": true}

Q: "Total spent on dining since 2024"
{"category": "dining", "start_year": 2024}

Q: "whats the total spend on donations from '23 to 2026?"
{"category": "donation", "start_year": 2023, "end_year": 2026}

Q: "Excluding rent, what was my total spend in 2025?"
{"year": 2025, "ignore_rent": true}"""
    },

    "get_top_expenses": {
        "parameters": "n (int, optional, defaults to 10), category (str, optional), year (int, optional), month (int, optional), day (int, optional), start_year (int, optional), start_month (int, optional), end_year (int, optional), end_month (int, optional), months (int, optional), min_amount (int, optional), ignore_rent (bool, optional, defaults to false)",
        "examples": """Q: "What were my biggest expenses in Dec 2024?"
{"n": 10, "year": 2024, "month": 12}

Q: "Top 5 food expenses in 2024?"
{"n": 5, "category": "Food", "year": 2024}

Q: "Top 10 food expenses in june 2025?"
{"n": 10, "category": "Food", "year": 2025, "month": 6}

Q: "Top 5 expenses for 26 june 2024?"
{"n": 5, "year": 2024, "month": 6, "day": 26}

Q: "Top expenses from July 2023 to Dec 2024"
{"start_year": 2023, "start_month": 7, "end_year": 2024, "end_month": 12}

Q: "Show my top 10 expenses from 2023 to 2025"
{"n": 10, "start_year": 2023, "end_year": 2025}

Q: "Top 10 expenses of past month?"
{"n": 10, "months": 1}

Q: "What are my top expenses excluding rent?"
{"n": 10, "ignore_rent": true}

Q: "Show my top 3 largest transactions in 2023"
{"n": 3, "year": 2023}

Q: "What were the biggest expenses over the last 3 months, without rent?"
{"n": 10, "months": 3, "ignore_rent": true}

Q: "Top 7 futsal expenses for 2025?"
{"n": 7, "category": "futsal game", "year": 2025}

Q: "Top 5 expenses last month, exclude rent"
{"n": 5, "months": 1, "ignore_rent": true}

Q: "What were my biggest expenses since 2023?"
{"n": 10, "start_year": 2023}

Q: "show me top 10 expenses from jan 2025 to june '25"
{"n": 10, "start_year": 2025, "start_month": 1, "end_year": 2025, "end_month": 6}

Q: "tell me the top 15 expenses for 2025, ignore rent plz"
{"n": 15, "year": 2025, "ignore_rent": true}"""
    }
}

def is_minicpm_model(model_id: str) -> bool:
    if not model_id:
        return False
    model_id_lower = model_id.lower()
    if "minicpm" in model_id_lower:
        return True
    
    # Try importing get_model_info dynamically to see if we can resolve the family registry
    try:
        from experiments.models import get_model_info
        info = get_model_info(model_id)
        if info:
            family = info.get("family", "")
            if family and "minicpm" in family.lower():
                return True
            m_id = info.get("id", "")
            if m_id and "minicpm" in m_id.lower():
                return True
    except Exception:
        pass
    return False


def get_tool_prompt(tool_name, model_id=None):
    if tool_name not in TOOL_PROMPTS:
        return None

    tool_data = TOOL_PROMPTS[tool_name]
    prompt = BASE_INSTRUCTIONS.format(
        metadata="{metadata}",
        current_date="{current_date}",
        parameters=tool_data["parameters"],
        examples=tool_data["examples"]
    )
    return prompt


def build_single_agent_prompt(metadata: str, current_date: str, model_id=None) -> str:
    import os
    import re
    base_dir = os.path.dirname(os.path.abspath(__file__))
    router_prompt_path = os.path.join(base_dir, "prompts", "router_prompt.txt")
    router_prompt_text = ""
    if os.path.exists(router_prompt_path):
        try:
            with open(router_prompt_path, "r", encoding="utf-8") as f:
                router_prompt_text = f.read()
        except Exception:
            pass

    tools_match = re.search(r"(## Available Tools.*?)(?=## Output Format|$)", router_prompt_text, re.DOTALL)
    available_tools_text = tools_match.group(1).strip() if tools_match else "## Available Tools\n1. plot_time_series (1)\n2. plot_distribution (2)\n3. plot_comparison_bars (3)\n4. calculate_total (4)\n5. get_top_expenses (5)"

    base_instructions_text = BASE_INSTRUCTIONS.replace(
        "{{\"category\": \"Food\", \"months\": 6}}",
        "{{\"tool\": 1, \"category\": \"Food\", \"months\": 6}}"
    )
    base_rules_match = re.search(r"(.*?)(?=## Available Parameters)", base_instructions_text, re.DOTALL)
    base_rules = base_rules_match.group(1).strip() if base_rules_match else base_instructions_text
    base_rules = base_rules.replace("{{", "{").replace("}}", "}")

    tool_map = {
        "plot_time_series": 1,
        "plot_distribution": 2,
        "plot_comparison_bars": 3,
        "calculate_total": 4,
        "get_top_expenses": 5
    }

    combined_examples = []
    parameters_text = []
    for tool_name, tool_id in tool_map.items():
        if tool_name in TOOL_PROMPTS:
            params = TOOL_PROMPTS[tool_name]["parameters"]
            parameters_text.append(f"{tool_id}. `{tool_name}`: {params}")
            
            examples_str = TOOL_PROMPTS[tool_name]["examples"]
            def replace_json(match):
                inner = match.group(1)
                if not inner.strip():
                    return f'{{"tool": {tool_id}}}'
                return f'{{"tool": {tool_id}, {inner}}}'
            
            modified_examples = re.sub(r'^\{([^}]*)\}$', replace_json, examples_str, flags=re.MULTILINE)
            combined_examples.append(f"### Tool {tool_id} ({tool_name}) Examples:\n{modified_examples}")

    all_examples = "\n\n".join(combined_examples)
    all_parameters = "\n".join(parameters_text)

    prompt = f"""{base_rules}

Your task is to determine which ONE tool is best suited to answer the user's question AND extract the required parameters.
You must output a single JSON object containing:
1. "tool": The numeric ID (1-5) of the best tool suited for the query.
2. The parameters for that tool.

{available_tools_text}

## PARAMETERS DEFINITIONS FOR EACH TOOL
{all_parameters}

## Examples
{all_examples}

## Context
```
{metadata}
```
Currency: JPY
Today: {current_date}

FINAL REMINDER: Output ONLY the JSON object. NO MARKDOWN, NO BACKTICKS, NO EXPLANATION, NO CODE!
"""
    return prompt