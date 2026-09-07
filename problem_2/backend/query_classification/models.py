from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class QueryClassification(BaseModel):
    query_type: Literal["semantic", "attributed", "temporal", "mixed"]
    sender: Optional[str] = None
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    original_query: str
