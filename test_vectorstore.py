from app.rag.embedding import create_embeddings
from app.rag.vectorstore import ChromaVectorStore


def main():
    # Test documents
    texts = [
        "Employees can apply for leave through the HR portal.",
        "The company provides health insurance to all full-time employees.",
        "Employees must complete their annual performance review by December."
    ]

    # Create embeddings
    embeddings = create_embeddings(texts)

    print(f"Created {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Create test chunks
    from app.rag.chunker import Result

    chunks = [
        Result(
            page_content=text,
            metadata={
                "source": f"test_document_{i}.md",
                "source_type": "document"
            }
        )
        for i, text in enumerate(texts)
    ]

    # Store in ChromaDB
    vectorstore = ChromaVectorStore()

    vectorstore.add_chunks(
        chunks,
        embeddings
    )

    print(f"ChromaDB contains {vectorstore.count()} chunks")


if __name__ == "__main__":
    main()