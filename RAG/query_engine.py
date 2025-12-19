from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from config import LLM_MODEL, RERANK_MODEL, DASHSCOPE_API_KEY
from typing import List

class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, vector_retriever):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        super().__init__()

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        bm25_nodes = self.bm25.retrieve(query)
        vector_nodes = self.vector.retrieve(query)
        # 合并 + 去重（按 id）
        seen_ids = set()
        combined = []
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in seen_ids:
                combined.append(n)
                seen_ids.add(n.node.node_id)
        return combined

def build_query_engine(bm25_retriever, vector_retriever):
    hybrid_retriever = HybridRetriever(bm25_retriever, vector_retriever)

    # 使用 DashScope 的 rerank 模型
    reranker = DashScopeRerank(
        api_key=DASHSCOPE_API_KEY,
        model=RERANK_MODEL,
        top_n=5
    )

    # 使用 DashScope LLM
    llm = DashScope(
        model_name=LLM_MODEL,
        api_key=DASHSCOPE_API_KEY
    )

    response_synthesizer = get_response_synthesizer(llm=llm)

    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker]
    )
    return query_engine