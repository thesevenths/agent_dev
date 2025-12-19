from llama_index.core import VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.dashscope import DashScopeEmbedding
from config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
import jieba

def get_bm25_retriever(nodes, top_k=5):
    # 中文分词适配
    def tokenize(text: str):
        return jieba.lcut(text)

    retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        tokenizer=tokenize,
        similarity_top_k=top_k
    )
    return retriever

def get_vector_retriever(nodes, top_k=5):
    embed_model = DashScopeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY
    )
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever