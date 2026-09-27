from pathlib import Path
from app.connectors.base import BaseConnector
from app.models.schemas import Document

#for fetching local documents
class LocalConnector(BaseConnector):

    def __init__(self, path: str):
        self.path = Path(path)

    def authenticate(self):
        return True

    def fetch_documents(self) -> list[Document]:

        documents = []

        for file in self.path.rglob("*.md"):

            text = file.read_text(encoding="utf-8")

            documents.append(
                Document(
                    text=text,
                    source=str(file),
                    source_type="document",
                    metadata={
                        "filename": file.name
                    }
                )
            )

        return documents