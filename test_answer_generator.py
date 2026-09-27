from app.rag.answer_generator import generate_answer


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
2023: Rating: 4.7/5

Exceptional performance with highest client satisfaction
scores in the team. Successfully expanded three key accounts.

Recognition: Customer Champion Award 2023 for highest NPS scores
""",
            "metadata": {
                "source": "Marcus Johnson.md"
            },
        },
    },
    {
        "subquery": (
            "Who received recognition for customer-related performance?"
        ),
        "result": {
            "document": """
2023: Rating: 4.7/5

Recognition: Customer Champion Award 2023 for highest NPS scores
""",
            "metadata": {
                "source": "Marcus Johnson.md"
            },
        },
    },
]


result = generate_answer(
    query,
    evidence,
)


print("\n")
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(result["answer"])


print("\n")
print("=" * 80)
print("SOURCES")
print("=" * 80)

for source in result["sources"]:
    print(source)