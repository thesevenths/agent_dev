from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import BaseRetriever  
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.llms.dashscope import DashScope
from config import LLM_MODEL, RERANK_MODEL, DASHSCOPE_API_KEY
from retrievers import filter_nodes_by_metadata
from typing import List, Optional
from llama_index.core.prompts import PromptTemplate

class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever, vector_retriever, metadata_filters: Optional[dict] = None):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.metadata_filters = metadata_filters or {}
        super().__init__()

    def _retrieve(self, query: str) -> List[NodeWithScore]:
        # 1. 向量检索（支持 filters）
        vector_nodes = self.vector.retrieve(query)
        
        # 2. BM25 检索 + 手动过滤
        bm25_nodes = self.bm25.retrieve(query)
        bm25_nodes = filter_nodes_by_metadata(bm25_nodes, self.metadata_filters)
        
        # 3. 合并去重
        seen_ids = set()
        combined = []
        for n in vector_nodes + bm25_nodes:
            if n.node.node_id not in seen_ids:
                # 可选：优先保留含表格的 chunk
                combined.append(n)
                seen_ids.add(n.node.node_id)
        return combined

def extract_filters_from_query(query: str) -> dict:
    """从 query 中提取年份等过滤条件（可扩展）"""
    filters = {}
    # 示例：提取年份
    import re
    year_match = re.search(r"(20\d{2})", query)
    if year_match:
        filters["fiscal_year"] = year_match.group(1)
    return filters

def build_query_engine(bm25_retriever, vector_retriever, raw_query: str):
    # 动态提取元数据过滤条件
    metadata_filters = extract_filters_from_query(raw_query)
    
    hybrid_retriever = HybridRetriever(
        bm25_retriever, 
        vector_retriever, 
        metadata_filters=metadata_filters
    )
    # print(f"DASHSCOPE_API_KEY:{DASHSCOPE_API_KEY}")
    # Rerank
    reranker = DashScopeRerank(
        api_key=DASHSCOPE_API_KEY,
        model=RERANK_MODEL,
        top_n=20
    )

    # LLM
    llm = DashScope(
        model_name=LLM_MODEL, 
        api_key=DASHSCOPE_API_KEY,
        max_tokens=8192,
        temperature=0.2,
        top_p=0.9,
        context_window=32768
    )

    system_prompt = PromptTemplate(
        "你是一位专业分析师，请基于以下检索到的上下文信息，撰写一份详尽、结构清晰、内容丰富的深度报告。\n"
        "报告应包括：背景介绍、核心发现、详细分析、案例支持、数据引用、潜在影响与建议。\n"
        "请确保内容不少于3000字（约4000+ tokens），语言专业、逻辑严密，避免重复。\n"
        "上下文信息如下：\n{context_str}\n"
        "用户问题：{query_str}\n"
        "markdown格式的深度报告："
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=system_prompt,
        response_mode="tree_summarize"
    )

    return RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[reranker]
    )