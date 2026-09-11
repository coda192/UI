from pydantic import BaseModel
from typing import List, Optional

class ColumnMetadata(BaseModel):
    name: str
    dtype: str
    missing_count: int

class DatasetMetadata(BaseModel):
    # DataSpec Static Metadata
    id: str
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
    local_data: Optional[bool] = None
    target: Optional[str] = None
    columns_to_use: Optional[List[str]] = None

    # Runtime / Computed Data Statistics (Optional if analysis hasn't run)
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: List[ColumnMetadata] = []
    identifier_columns: List[str] = []
    compatible_tasks: List[str] = []


class ColumnProfile(BaseModel):
    name: str
    detected_type: str
    dtype: Optional[str] = None
    missing_count: int
    missing_percentage: float


class DatasetProfileResponse(BaseModel):
    dataset_id: str
    row_count: int
    column_count: int
    memory_usage_mb: float
    columns_with_missing: int
    total_missing_values: int
    columns: List[ColumnProfile]

