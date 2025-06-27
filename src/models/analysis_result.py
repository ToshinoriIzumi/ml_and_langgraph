from typing import Any, Optional

from langchain_core.pydantic_v1 import BaseModel, Field, ConfigDict

class AnalysisResult(BaseModel):
    """
    Analysis result model
    """
    model: Any = Field(description="The model of the analysis result")
    mse: Optional[float] = Field(description="The mean squared error of the analysis result")
    accuracy: Optional[float] = Field(description="The accuracy of the analysis result")

    class Config:
        arbitrary_types_allowed = True