from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from models.analysis_data import AnalysisData

@tool
def load_data(file_path: str) -> pd.DataFrame:
    """指定されたファイルパスからデータを読み込みます。
    
    Args:
        file_path (str): 読み込むファイルのパス
        
    Returns:
        pd.DataFrame: 読み込まれたデータフレーム
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

class AnalysisDataLoader:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # ツールをLLMにバインド
        self.llm_with_tools = self.llm.bind_tools([load_data])
        
    def load(self, prompt: str) -> AnalysisData:
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system", 
                "あなたは、データサイエンティストです。機械学習モデルを作成するために必要なデータを生成します。"
                "必要に応じてload_dataツールを使用してファイルからデータを読み込んでください。"
            ),
            (
                "user",
                """以下の手順に従って処理を行ってください。
                1. {prompt}に含まれているファイルパスを抽出する。
                2. 抽出したファイルパスからデータを読み込む。(load_dataツールを使用してください)
                
                プロンプト: {prompt}
                """
            )
        ])
        
        # LLMにメッセージを送信
        formatted_prompt = prompt_template.format_messages(prompt=prompt)
        response = self.llm_with_tools.invoke(formatted_prompt)
        
        # ツール呼び出しを処理
        data_df = None
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call["name"] == "load_data":
                    # ツールを実行してDataFrameを取得
                    data_df = load_data.invoke(tool_call["args"])
                    print(f"データを読み込みました: {data_df.shape}")
                    print(f"カラム: {list(data_df.columns)}")
                    break
        else:
            raise ValueError("データの読み込みに失敗しました。")
        
        if data_df is not None:
            try:
                return self._create_analysis_data(data_df)
            except Exception as e:
                raise ValueError(f"データの分割処理中にエラーが発生しました: {e}")
        
        raise ValueError("データの読み込みに失敗しました。")

    def _create_analysis_data(self, data_df: pd.DataFrame) -> AnalysisData:
        X_train, X_test, y_train, y_test = train_test_split(
            data_df.drop(columns=["target"]),
            data_df["target"],
            test_size=0.2,
            random_state=42
        )
        training_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        return AnalysisData(
            training_data=training_data,
            test_data=test_data,
            target_column="target",
            features=X_train.columns.tolist()
        )


if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    loader = AnalysisDataLoader(llm)
    analysis_data = loader.load("""
    以下のファイルパスからデータを読み込んで、機械学習用のデータを生成してください。
    ファイルパス: /app/data/iris.csv
    """)
    print(analysis_data)