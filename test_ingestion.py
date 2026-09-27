# from app.services.ingestion import ingest_local_documents


# if __name__ == "__main__":
#     ingest_local_documents("../knowledge-base")

from app.connectors.local import LocalConnector
from app.rag.chunker import process_document


def main():
    connector = LocalConnector("../knowledge-base/employees/test-employee")

    documents = connector.fetch_documents()

    print(f"Loaded {len(documents)} documents")

    document = next(
        doc for doc in documents
        if "David Kim.md" in doc.source
    )

    print(f"\nTesting document: {document.source}")

    chunks = process_document(document)

    print(f"\nCreated {len(chunks)} chunks")

    for i, chunk in enumerate(chunks, start=1):
        print(f"\n===== CHUNK {i} =====")
        print(chunk.page_content)


if __name__ == "__main__":
    main()