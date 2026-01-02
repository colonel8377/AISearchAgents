"""Few-shot examples for Web Opinion Extractor atomization."""

ATOMIZER_FEW_SHOTS = """You are an expert text analyzer. Your task is to split complex text into atomic units.

INSTRUCTIONS:
1. Break down compound sentences into separate atomic units
2. Each atomic unit should express ONE and only ONE statement
3. Classify each unit as either "fact" or "opinion"
   - FACT: Verifiable, objective statement (e.g., "The bill was passed on January 5th")
   - OPINION: Subjective viewpoint, belief, or judgment (e.g., "The policy is harmful")

EXAMPLES:

Input: "I support the tax cut but oppose the trade war."
Output:
[
  {"statement": "Support for the tax cut", "type": "opinion", "original_sentence": "I support the tax cut but oppose the trade war."},
  {"statement": "Opposition to the trade war", "type": "opinion", "original_sentence": "I support the tax cut but oppose the trade war."}
]

Input: "The unemployment rate fell to 3.5% in October, which shows excellent economic management."
Output:
[
  {"statement": "Unemployment rate fell to 3.5% in October", "type": "fact", "original_sentence": "The unemployment rate fell to 3.5% in October, which shows excellent economic management."},
  {"statement": "This shows excellent economic management", "type": "opinion", "original_sentence": "The unemployment rate fell to 3.5% in October, which shows excellent economic management."}
]

Return ONLY a JSON array of atomic units."""
