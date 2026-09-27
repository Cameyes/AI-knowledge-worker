from app.rag.query_decomposer import decompose_query


query = (
    "Which employee had a 2023 performance rating of 4.7/5 "
    "and also received recognition for customer-related performance?"
)


subqueries = decompose_query(query)


print("\nGenerated subqueries:")

for i, subquery in enumerate(subqueries, start=1):
    print(f"{i}. {subquery}")