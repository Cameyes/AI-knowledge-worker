from pathlib import Path
import chromadb
from app.rag.chunker import Result

DB_PATH = str(
    Path(__file__).resolve().parents[3] / "chroma_db"
)

COLLECTION_NAME = "docs"

class ChromaVectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    def add_chunks( self,chunks: list[Result],embeddings: list[list[float]]):
        start_id = self.collection.count()

        ids = [str(start_id + i) for i in range(len(chunks))]

        documents = [chunk.page_content for chunk in chunks]

        metadatas = [ chunk.metadata for chunk in chunks]

        self.collection.add(ids=ids,documents=documents, embeddings=embeddings,metadatas=metadatas,)

    def count(self):
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )