import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field, ConfigDict

class AnalysisData(BaseModel):
    """
    Analysis data model
    """
    training_data: pd.DataFrame = Field(description="The training data")
    test_data: pd.DataFrame = Field(description="The test data")
    target_column: str = Field(description="The target column")
    features: list[str] = Field(description="The features")

    class Config:
        arbitrary_types_allowed = True