from app.rag.evidence_retriever import multi_evidence_retrieve
from app.rag.evidence_merger import merge_evidence
from app.rag.evidence_verifier import verify_evidence
from app.rag.answer_generator import generate_answer



QUESTIONS = [
    "What were the 2022 bonuses of David Kim, Kevin Zhang, Priya Sharma, Robert Chen and Carlos Rodriguez?"
    # "How did David Kim's base salary change from 2020 to 2023, and what was his bonus in 2023?",
    # "Which certifications does David Kim already hold, what certification is he currently pursuing, and which of his current responsibilities are related to AWS and Kubernetes?",
]


def run_pipeline(query: str):
    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print("\n")
    print("=" * 80)
    print("1. RETRIEVING EVIDENCE")
    print("=" * 80)

    evidence = multi_evidence_retrieve(
        query,
        retrieval_top_k=30,
        rerank_top_k=10,
    )

    print("\n")
    print("=" * 80)
    print("2. MERGING EVIDENCE")
    print("=" * 80)

    merged_evidence = merge_evidence(evidence)

    print(f"\nMerged evidence items: {len(merged_evidence)}")

    print("\n===== MERGED EVIDENCE BY SUBQUERY =====")

    for i, item in enumerate(merged_evidence, 1):
        print(
            f"{i}. "
            f"{item['subquery']} | "
            f"{item['result']['metadata'].get('source')}"
        )

    print("\n")
    print("=" * 80)
    print("3. VERIFYING EVIDENCE")
    print("=" * 80)

    verified_evidence = verify_evidence(
        query,
        merged_evidence,
    )

    for i, e in enumerate(verified_evidence, start=1):
        print(f"\n===== VERIFIED EVIDENCE {i} =====")
        print("Subquery:", e["subquery"])
        print("Source:", e["result"]["metadata"].get("source"))
        print("Evidence:")
        print(e["result"]["document"])
        print(f"\nVerified evidence items: {len(verified_evidence)}")

    print("\n")
    print("=" * 80)
    print("4. GENERATING ANSWER")
    print("=" * 80)

    generation_evidence = {}

    for item in verified_evidence:
        source = item["result"]["metadata"].get(
            "source",
            "Unknown source",
        )

    generation_evidence.setdefault(source, []).append(item)

    result = generate_answer(
        query,
        generation_evidence,
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


def main():
    for question in QUESTIONS:
        run_pipeline(question)


if __name__ == "__main__":
    main()