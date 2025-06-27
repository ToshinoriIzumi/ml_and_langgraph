from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from analysis.analysis_data_loader import AnalysisDataLoader
from analysis.ml_analysis import MlAnalysis
from models.analysis_state import AnalysisState

class AnalysisAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.graph = self._build_graph()

    def run(self, prompt: str) -> str:
        initial_state = AnalysisState(prompt=prompt)
        result = self.graph.invoke(initial_state)
        return result["final_result"]
        
    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(AnalysisState)
        workflow.add_node("load_data", self._load_data)
        workflow.add_node("execute_ml_analysis", self._execute_ml_analysis)
        workflow.add_node("check_sufficient", self._check_sufficient)
        workflow.add_node("generate_final_result", self._generate_final_result)
        workflow.set_entry_point("load_data")
        workflow.add_edge("load_data", "execute_ml_analysis")
        workflow.add_edge("execute_ml_analysis", "check_sufficient")
        workflow.add_edge("check_sufficient", "generate_final_result")
        workflow.add_edge("generate_final_result", END)
        return workflow.compile()
        
    def _load_data(self, state: AnalysisState) -> Dict[str, Any]:
        data_loader = AnalysisDataLoader(self.llm)
        analysis_data = data_loader.load(state.prompt)    
        return {"analysis_data": analysis_data}
    
    def _execute_ml_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        ml_analysis = MlAnalysis(self.llm)
        analysis_result = ml_analysis.execute(state.prompt, state.analysis_data)
        return {"analysis_result": analysis_result}
    
    def _check_sufficient(self, state: AnalysisState) -> Dict[str, Any]:
        if state.analysis_result.mse is not None and state.analysis_result.mse < 0.1:
            return {"is_sufficient": True}
        elif state.analysis_result.accuracy is not None and state.analysis_result.accuracy > 0.9:
            return {"is_sufficient": True}
        else:
            return {"is_sufficient": False}
        
    def _generate_final_result(self, state: AnalysisState) -> Dict[str, Any]:
        if state.analysis_result.mse is not None:
            result = f"MSE: {state.analysis_result.mse}"
        elif state.analysis_result.accuracy is not None:
            result = f"Accuracy: {state.analysis_result.accuracy}"
        else:
            result = "解析結果が不足しています。"
        
        return {"final_result": result}

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = AnalysisAgent(llm)
    result = agent.run("次のデータセットを使用して、解析を行ってください。データセットのパス: data/iris.csv")
    print(result)
    
if __name__ == "__main__":
    main()