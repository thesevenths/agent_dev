from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter
from config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
import jieba

def get_bm25_retriever(nodes, top_k=5, filters=None):
    def tokenize(text: str):
        return jieba.lcut(text)
    
    return BM25Retriever.from_defaults(
        nodes=nodes,
        tokenizer=tokenize,
        similarity_top_k=top_k,
        filters=filters  # 注意：BM25Retriever 0.14.10 可能不支持 filters，我们将在 HybridRetriever 中后过滤
    )

def get_vector_retriever(nodes, top_k=5, filters=None):
    embed_model = DashScopeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY
    )
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    return index.as_retriever(similarity_top_k=top_k, filters=filters)

# 为兼容性，我们在 HybridRetriever 中手动过滤 BM25 结果
def filter_nodes_by_metadata(nodes, filters_dict):
    """简易元数据过滤（用于 BM25）"""
    if not filters_dict:
        return nodes
    filtered = []
    for node in nodes:
        match = True
        for key, value in filters_dict.items():
            if node.metadata.get(key) != value:
                match = False
                break
        if match:
            filtered.append(node)
    return filtered