from langchain_core.pydantic_v1 import BaseModel, Field

class EvaluationResults(BaseModel):
    is_sufficient: bool = Field(description="解析結果は十分かどうか")
    reason: str = Field(description="判断の理由")