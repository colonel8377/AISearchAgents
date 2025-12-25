"""Few-shot examples for Compare Claims Agent."""

COMPARE_CLAIMS_FEW_SHOTS = """EXAMPLES:

Example 1:
SUMMARY CLAIM: "The company reported record profits this quarter"
URL CLAIM: "Q3 earnings exceeded all previous quarters with $2.1 billion in revenue"
STATUS: supported
CONFIDENCE: 0.95
REASON: The URL claim provides specific evidence (Q3 earnings, $2.1 billion) that directly supports the summary's statement about record profits.

Example 2:
SUMMARY CLAIM: "The policy will reduce unemployment"
URL CLAIM: "The new policy is expected to increase unemployment by 2%"
STATUS: contradicted
CONFIDENCE: 0.90
REASON: The URL claim explicitly states the opposite - unemployment will increase, not decrease.

Example 3:
SUMMARY CLAIM: "The meeting discussed budget allocations"
URL CLAIM: "The meeting covered various topics including staffing and resources"
STATUS: neutral
CONFIDENCE: 0.60
REASON: The URL claim mentions topics related to budget (staffing, resources) but doesn't specifically confirm budget allocations were discussed.

"""



