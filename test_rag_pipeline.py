from app.rag.evidence_retriever import multi_evidence_retrieve
from app.rag.evidence_merger import merge_evidence
from app.rag.evidence_verifier import verify_evidence
from app.rag.answer_generator import generate_answer
import time

TEST_CASES = [
    {
        "question": "Which employee had a 2023 performance rating of 4.7/5?",
        "expected": "Lisa Anderson",
    },
    {
        "question": "Who won the IIOTY award?",
        "expected": "Maxine Thompson",
    },
    {
        "question": "Who is a Customer Success Manager?",
        "expected": "Marcus Johnson",
    },
    {
        "question": "Which employee has the highest current salary?",
        "expected": "James Wilson",
    },
    {
        "question": "Which employee received the Customer Champion Award 2023?",
        "expected": "Marcus Johnson",
    },
    {
        "question": "Which employee has AWS Solutions Architect Professional certification?",
        "expected": "Carlos Rodriguez",
    },
    {
        "question": "Which employee had a 2023 rating of 4.9/5 and works in engineering?",
        "expected": "James Wilson",
    },
    {
        "question": (
            "Which employee had a 2023 performance rating of 4.7/5 "
            "and also received recognition for customer-related performance?"
        ),
        "expected": "Marcus Johnson",
    },
    {
        "question": "Who manages a portfolio of 25 enterprise clients?",
        "expected": "Marcus Johnson",
    },
    {
        "question": (
            "Which employee received recognition for customer-related "
            "performance and had a 2023 performance rating of 4.7/5?"
        ),
        "expected": "Marcus Johnson",
    },
]


def main():

    passed = 0

    print("\n")
    print("=" * 80)
    print("RAG EVALUATION")
    print("=" * 80)

    for index, test_case in enumerate(TEST_CASES, start=1):

        question = test_case["question"]
        expected = test_case["expected"]

        print("\n")
        print("=" * 80)
        print(f"TEST CASE #{index}")
        print("=" * 80)

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected}")

        try:

            evidence = multi_evidence_retrieve(
                question,
                retrieval_top_k=10,
                rerank_top_k=5,
            )

            merged_evidence = merge_evidence(
                evidence
            )

            verified_evidence = verify_evidence(
                question,
                merged_evidence,
            )

            result = generate_answer(
                question,
                verified_evidence,
            )

            answer = result["answer"]

            print(f"Actual:   {answer}")

            if expected.lower() in answer.lower():

                print("Result:   PASS")
                passed += 1

            else:

                print("Result:   FAIL")

        except Exception as e:

            print(f"Result:   ERROR")
            print(f"Error:    {e}")

        time.sleep(3)
        
    print("\n")
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    print(f"Passed: {passed}/{len(TEST_CASES)}")

    if passed == len(TEST_CASES):
        print("Result: 10/10 PASS")
    else:
        print("Result: Evaluation needs investigation")


if __name__ == "__main__":
    main()