from app.rag.evidence_verifier import verify_evidence


query = (
    "Which employee had a 2023 performance rating of 4.7/5 "
    "and also received recognition for customer-related performance?"
)


evidence = [
    {
        "subquery": (
            "Which employee had a 2023 performance rating of 4.7/5?"
        ),
        "result": {
            "document": """
2023: Rating: 4.5/5

Strong performance. Successfully led diversity hiring initiative.
""",
            "metadata": {
                "source": "Amanda Foster.md"
            },
        },
    },
    {
        "subquery": (
            "Which employee had a 2023 performance rating of 4.7/5?"
        ),
        "result": {
            "document": """
2023: Rating: 4.7/5

Exceptional performance with highest client satisfaction
scores in the team. Successfully expanded three key accounts.
""",
            "metadata": {
                "source": "Marcus Johnson.md"
            },
        },
    },
    {
        "subquery": (
            "Which employee received recognition for customer-related performance?"
        ),
        "result": {
            "document": """
Recognition: Customer Champion Award 2023 for highest NPS scores.
""",
            "metadata": {
                "source": "Marcus Johnson.md"
            },
        },
    },
    {
        "subquery": (
            "Which employee received recognition for customer-related performance?"
        ),
        "result": {
            "document": """
2023: Rating: 3.3/5

Needs to improve empathy in customer interactions.
""",
            "metadata": {
                "source": "Brandon Walker.md"
            },
        },
    },
]


verified_evidence = verify_evidence(
    query,
    evidence,
)


print("\n")
print("=" * 80)
print("VERIFIED EVIDENCE")
print("=" * 80)


for i, item in enumerate(verified_evidence, start=1):
    print("\n")
    print("-" * 80)
    print(f"EVIDENCE #{i}")
    print("-" * 80)

    print(f"Subquery: {item['subquery']}")
    print(f"Source: {item['result']['metadata']['source']}")

    print("\nDocument:")
    print(item["result"]["document"])