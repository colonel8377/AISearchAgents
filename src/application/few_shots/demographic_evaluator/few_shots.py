"""Optimized few-shot examples for Demographic Evaluator Agent."""

DEMOGRAPHIC_EVALUATOR_FEW_SHOTS = """Here are concise examples of evaluating sentences from different demographic perspectives:

EXAMPLE 1 - Progressive Social Worker:
Profile: 28-year-old female, college-educated, urban social worker, progressive, middle-class, African American background.

Sentences:
1. "Universal healthcare should be a fundamental right."
2. "Corporate tax cuts stimulate economic growth."

Output:
{
  "judgments": [
    {"index": 0, "sentence": "Universal healthcare should be a fundamental right.", "agree": 1, "reason": "I've seen clients avoid medical care due to costs. Healthcare is a basic human need that everyone deserves access to."},
    {"index": 1, "sentence": "Corporate tax cuts stimulate economic growth.", "agree": 0, "reason": "Tax breaks for corporations rarely help working families. We need policies that directly support people struggling to make ends meet."}
  ]
}

EXAMPLE 2 - Conservative Business Owner:
Profile: 55-year-old male, high school education, rural small business owner, conservative, middle-class, white background.

Sentences:
1. "Government regulations hurt small businesses."
2. "Second Amendment rights must be protected."

Output:
{
  "judgments": [
    {"index": 0, "sentence": "Government regulations hurt small businesses.", "agree": 1, "reason": "Running a small construction company, the paperwork and compliance costs are overwhelming. Big corporations handle regulations better than we can."},
    {"index": 1, "sentence": "Second Amendment rights must be protected.", "agree": 1, "reason": "Out here in rural areas, we rely on firearms for protection and hunting. Gun rights are part of our way of life and ability to protect ourselves."}
  ]
}

EXAMPLE 3 - Independent Engineer:
Profile: 42-year-old non-binary, graduate degree, suburban engineer, independent, upper-middle class, Asian American background.

Sentences:
1. "STEM education should receive more funding."
2. "Social media platforms need better moderation."

Output:
{
  "judgments": [
    {"index": 0, "sentence": "STEM education should receive more funding.", "agree": 1, "reason": "Working in tech, I see the huge demand for STEM skills. Early investment in STEM education is crucial for future competitiveness."},
    {"index": 1, "sentence": "Social media platforms need better moderation.", "agree": 1, "reason": "Free speech matters, but misinformation spreads harm. We need thoughtful moderation that protects users while preserving open discourse."}
  ]
}

GUIDELINES:
- Speak naturally from the person's perspective using their experiences and values
- NEVER mention demographic traits (age, gender, etc.) in reasoning
- Show how background and values influence judgment
- Be authentic and personal, not generic
- Consider the person's likely life circumstances
"""

