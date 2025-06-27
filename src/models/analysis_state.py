from typing import Optional, Dict, Any
from langchain_core.pydantic_v1 import BaseModel, Field
from models.analysis_data import AnalysisData
from models.analysis_result import AnalysisResult


class AnalysisState(BaseModel):
    """
    Analysis state model
    """
    prompt: str = Field(description="ユーザーの入力")
    analysis_data: AnalysisData = Field(description="解析用のデータ", default=None)
    analysis_result: AnalysisResult = Field(description="解析結果", default=None)
    is_sufficient: bool = Field(description="解析結果は十分かどうか", default=None)
    final_result: str = Field(description="生成された応答", default=None)