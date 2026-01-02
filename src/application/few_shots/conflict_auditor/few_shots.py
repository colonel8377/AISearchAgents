"""Few-shot examples for Conflict Auditor Agent."""

CONFLICT_AUDITOR_FEW_SHOTS = """Example 1 - Numerical Discrepancy:
CLAIM_1: The company reported .1 billion in revenue for Q3 2023.
EVIDENCE:
  - "The company reported .3 billion in revenue for Q3 2023, representing a 15% increase."

-- CLAIM_ID: 1
-- VERDICT: Contradicted
-- CONFLICT_TYPE: Numerical discrepancy
- ANALYSIS: The claim states .1 billion but evidence shows .3 billion, indicating a numerical conflict.

Example 2 - Supported:
CLAIM_2: The revenue increased by 15%.
EVIDENCE:
  - "The company reported .3 billion in revenue for Q3 2023, representing a 15% increase."

-- CLAIM_ID: 2
-- VERDICT: Supported
-- CONFLICT_TYPE: N/A
- ANALYSIS: The evidence explicitly confirms the 15% increase figure.

Example 3 - Missing Evidence:
CLAIM_3: The CEO is named John Smith.
EVIDENCE: NO_EVIDENCE_FOUND

-- CLAIM_ID: 3
-- VERDICT: Neutral
-- CONFLICT_TYPE: Missing evidence
- ANALYSIS: No evidence was found in the text to either confirm or contradict the CEO's name."""

