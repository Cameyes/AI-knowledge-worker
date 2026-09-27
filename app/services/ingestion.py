from app.connectors.local import LocalConnector
from app.rag.chunker import create_chunks
from app.rag.local_chunker import create_local_chunks
from app.rag.embedding import create_embeddings
from app.rag.vectorstore import ChromaVectorStore


def ingest_local_documents(path: str):

    # 1. Fetch documents
    connector = LocalConnector(path)
    documents = connector.fetch_documents()

    print(f"Loaded {len(documents)} documents")

    # 2. Convert documents into chunks
    # chunks = create_chunks(documents)
    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks")

    # 3. Create embeddings for each chunk
    texts = [chunk.page_content for chunk in chunks]

    embeddings = create_embeddings(texts)

    print(f"Created {len(embeddings)} embeddings")

    # 4. Store chunks + embeddings in ChromaDB
    vectorstore = ChromaVectorStore()

    vectorstore.reset()

    vectorstore.add_chunks(chunks, embeddings)

    print(
        f"ChromaDB now contains "
        f"{vectorstore.count()} chunks"
    )

if __name__ == "__main__":
    ingest_local_documents("../knowledge-base/employees/")