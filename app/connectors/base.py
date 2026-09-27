from abc import ABC, abstractmethod
from app.models.schemas import Document

#any connector we create must provide these two functions - generic connector schema
class BaseConnector(ABC):

    @abstractmethod
    def authenticate(self):
        pass

    @abstractmethod
    def fetch_documents(self) -> list[Document]:
        pass