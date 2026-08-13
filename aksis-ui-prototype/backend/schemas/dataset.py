from pydantic import BaseModel
from typing import List, Optional

class ColumnMetadata(BaseModel):
    name: str
    dtype: str
    missing_count: int

class DatasetMetadata(BaseModel):
    id: str
    name: str
    row_count: int
    column_count: int
    columns: List[ColumnMetadata]
    target: Optional[str] = None
    identifier_columns: List[str] = []
    compatible_tasks: List[str] = []
