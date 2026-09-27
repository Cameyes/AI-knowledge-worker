from app.rag.embedding import create_embeddings
from app.rag.vectorstore import ChromaVectorStore


def retrieve(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
):
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:
            User query or retrieval subquery.

        top_k:
            Maximum number of chunks to retrieve.

        filters:
            Optional metadata filters for restricting retrieval.

    Returns:
        A list of retrieved results containing:
        - document
        - metadata
        - distance
    """

    # ---------------------------------------------------------
    # 1. Convert the query into an embedding
    # ---------------------------------------------------------

    # query_embedding = create_embeddings(
    #     [query]
    # )[0]
    #create embedding function changes according to the encoding model we choose
    query_embedding = create_embeddings(
    [query],
    is_query=True,
)[0]

    # ---------------------------------------------------------
    # 2. Access the vector store
    # ---------------------------------------------------------

    vectorstore = ChromaVectorStore()

    # ---------------------------------------------------------
    # 3. Build query arguments
    # ---------------------------------------------------------

    query_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    # Add metadata filtering only when requested.
    if filters:
        query_arguments["where"] = filters

    # ---------------------------------------------------------
    # 4. Search ChromaDB
    # ---------------------------------------------------------

    results = vectorstore.collection.query(
        **query_arguments
    )

    # ---------------------------------------------------------
    # 5. Convert ChromaDB response into our
    #    application-level result format
    # ---------------------------------------------------------

    retrieved_results = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved_results.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_results

def retrieve_by_source(
    query: str,
    source: str,
    top_k: int = 10,
    filters: dict | None = None,
):
    """
    Retrieve the most relevant chunks for a query,
    restricted to a specific source.

    This enables generic source-local semantic expansion.

    Returns:
        A list of chunks containing:
        - document
        - metadata
        - distance
        - local_retrieval_rank
    """

    # ---------------------------------------------------------
    # 1. Create query embedding
    # ---------------------------------------------------------

    query_embedding = create_embeddings(
        [query],
        is_query=True,
    )[0]

    # ---------------------------------------------------------
    # 2. Access vector store
    # ---------------------------------------------------------

    vectorstore = ChromaVectorStore()

    # ---------------------------------------------------------
    # 3. Build metadata filter
    # ---------------------------------------------------------

    where = {
        "filename": source,
    }

    if filters:
        where = {
            "$and": [
                {"filename": source},
                filters,
            ]
        }

    # ---------------------------------------------------------
    # 4. Query Chroma semantically within this source
    # ---------------------------------------------------------

    results = vectorstore.collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )

    # ---------------------------------------------------------
    # 5. Convert to application result format
    # ---------------------------------------------------------

    retrieved_results = []

    for local_rank, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        retrieved_results.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "local_retrieval_rank": local_rank,
            }
        )

    return retrieved_results