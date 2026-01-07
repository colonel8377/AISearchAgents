"""Few-shot examples for Consistency Agent."""

CONSISTENCY_FEW_SHOTS = """Example 1 - Supported Claims:
SUMMARY CLAIM: The company reported $2.3 billion in revenue for Q3 2023.
URL CLAIM: The company's Q3 2023 revenue reached $2.3 billion, representing a 15% increase.

-- STATUS: supported
-- CONFIDENCE: 0.95
-- REASON: The URL claim explicitly confirms the revenue figure stated in the summary claim.

Example 2 - Contradicted Claims:
SUMMARY CLAIM: The company reported $2.1 billion in revenue for Q3 2023.
URL CLAIM: The company's Q3 2023 revenue reached $2.3 billion, representing a 15% increase.

-- STATUS: contradicted
-- CONFIDENCE: 0.90
-- REASON: The summary claim states $2.1 billion but the URL claim shows $2.3 billion, indicating a numerical discrepancy.

Example 3 - Neutral Claims:
SUMMARY CLAIM: The company is planning to expand into new markets.
URL CLAIM: The company reported $2.3 billion in revenue for Q3 2023.

-- STATUS: neutral
-- CONFIDENCE: 0.50
-- REASON: The URL claim provides revenue information but does not address the expansion plans mentioned in the summary claim.
"""

