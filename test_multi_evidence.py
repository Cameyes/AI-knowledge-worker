from app.rag.evidence_retriever import multi_evidence_retrieve
from app.rag.evidence_merger import merge_evidence


query = (
    "Which employee had a 2023 performance rating of 4.7/5 "
    "and also received recognition for customer-related performance?"
)


evidence = multi_evidence_retrieve(query)

merged_evidence = merge_evidence(evidence)


print("\n")
print("=" * 80)
print("MERGED EVIDENCE")
print("=" * 80)


for i, item in enumerate(merged_evidence, start=1):

    print("\n")
    print("-" * 80)
    print(f"EVIDENCE #{i}")
    print("-" * 80)

    print(f"Subquery: {item['subquery']}")

    result = item["result"]

    print(f"Distance: {result['distance']}")
    print(f"Source: {result['metadata']['source']}")

    print("\nDocument:")
    print(result["document"])