import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models.analysis_result import AnalysisResult
from models.analysis_data import AnalysisData

class MlAnalysis:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    def execute(self, prompt: str, analysis_data: AnalysisData) -> AnalysisResult:
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system", 
                """
                あなたは機械学習の専門家です。
                プロンプトから適切な機械学習モデルを選択してください。
                モデルの選択肢は以下の通りです。
                - 線形回帰
                - ロジスティック回帰
                - 決定木(回帰)
                - 決定木(分類)
                - ランダムフォレスト(回帰)
                - ランダムフォレスト(分類)
                - サポートベクターマシン(回帰)
                - サポートベクターマシン(分類)
                - 勾配ブースティング決定木(回帰)
                - 勾配ブースティング決定木(分類)
                結果のみを出力してください。
                
                モデルの選択には、以下の情報を使用してください。
                - データセット: {training_data}
                - 特徴量: {features}
                - 目的変数: {target_column}
                """
            ),
            (
                "user",
                """
                プロンプト: {prompt}
                """
            )
        ])
        
        
        chain = prompt_template | self.llm | StrOutputParser()
        result = chain.invoke({
            "training_data": str(analysis_data.training_data),
            "features": str(analysis_data.features),
            "target_column": str(analysis_data.target_column),
            "prompt": prompt
        })

        if result == "線形回帰":
            model = LinearRegression()
            model.fit(analysis_data.training_data[analysis_data.features], analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            mse = mean_squared_error(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                mse=mse
            )
        elif result == "ロジスティック回帰":
            model = LogisticRegression()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            accuracy = accuracy_score(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                accuracy=accuracy
            )
        elif result == "決定木(回帰)":
            model = DecisionTreeRegressor()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            mse = mean_squared_error(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                mse=mse
            )
        elif result == "ランダムフォレスト(回帰)":
            model = RandomForestRegressor()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features]) 
            mse = mean_squared_error(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                mse=mse
            )
        elif result == "サポートベクターマシン(回帰)":
            model = SVC()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            accuracy = accuracy_score(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                accuracy=accuracy
            )
        elif result == "勾配ブースティング決定木(回帰)":
            model = GradientBoostingRegressor()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            mse = mean_squared_error(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                mse=mse
            )
        elif result == "決定木(分類)":
            model = DecisionTreeClassifier()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            accuracy = accuracy_score(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                accuracy=accuracy
            )
        elif result == "ランダムフォレスト(分類)":
            model = RandomForestClassifier()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            accuracy = accuracy_score(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                accuracy=accuracy
            )
        elif result == "サポートベクターマシン(分類)":
            model = SVC()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            accuracy = accuracy_score(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                accuracy=accuracy
            )
        elif result == "勾配ブースティング決定木(分類)":
            model = GradientBoostingClassifier()
            model.fit(analysis_data.training_data[analysis_data.features],
                      analysis_data.training_data[analysis_data.target_column])
            y_pred = model.predict(analysis_data.test_data[analysis_data.features])
            accuracy = accuracy_score(analysis_data.test_data[analysis_data.target_column], y_pred)
            return AnalysisResult(
                model=model,
                accuracy=accuracy
            )
        else:
            raise ValueError("モデルの選択に失敗しました。")
        

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    df = pd.read_csv("data/iris.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1),
                                                        df["target"],
                                                        test_size=0.2,
                                                        random_state=42)
    
    analysis_data = AnalysisData(
        training_data=pd.concat([X_train, y_train], axis=1),
        test_data=pd.concat([X_test, y_test], axis=1),
        features=X_train.columns.tolist(),
        target_column="target"
    )
    ml_analysis = MlAnalysis(llm)
    result = ml_analysis.execute("データセットの予測を行ってください。", analysis_data)
    print(result)