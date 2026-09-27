# from sentence_transformers import SentenceTransformer
# from app.config import EMBEDDING_MODEL

# model = SentenceTransformer(EMBEDDING_MODEL)

# def create_embeddings(texts: list[str]) -> list[list[float]]:
#     embeddings = model.encode(texts,show_progress_bar=True )
#     return embeddings.tolist()


from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

model = SentenceTransformer(
    EMBEDDING_MODEL,
    trust_remote_code=True,
)

def create_embeddings(
    texts: list[str],
    is_query: bool = False,
) -> list[list[float]]:
    if is_query:
        embeddings = model.encode(
            texts,
            prompt_name="query",
            show_progress_bar=True,
        )
    else:
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
        )

    return embeddings.tolist()