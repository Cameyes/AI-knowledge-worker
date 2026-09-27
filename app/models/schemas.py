from pydantic import BaseModel
from typing import Any
from pydantic import BaseModel

#creating generic structure for our document
class Document(BaseModel):
    text: str
    source: str
    source_type: str
    metadata: dict[str, Any] = {}

class Result(BaseModel):
    page_content: str
    metadata: dict