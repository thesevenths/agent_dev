from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, ExactMatchFilter
from config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
import jieba
import chromadb
from chromadb.errors import NotFoundError
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

CHROMA_PATH = "E:\\model\\RAG\\chroma_db"

def get_bm25_retriever(nodes, top_k=5, filters=None):
    def tokenize(text: str):
        return jieba.lcut(text)
    
    return BM25Retriever.from_defaults(
        nodes=nodes,
        tokenizer=tokenize,
        similarity_top_k=top_k,
        filters=filters  # 注意：BM25Retriever 0.14.10 可能不支持 filters，将在 HybridRetriever 中后过滤
    )

def get_vector_retriever(nodes=None, top_k=5, filters=None, persist_dir=CHROMA_PATH):
    """
    如果 persist_dir 存在且非空，则加载已有索引；
    否则用 nodes 构建新索引并持久化。
    """
    embed_model = DashScopeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        batch_size=10  # 必须保留！
    )

    # 初始化 Chroma 客户端（持久化）
    db = chromadb.PersistentClient(path=persist_dir)
    
    # 检查是否已有 collection
    collection_name = "annual_reports"
    try:
        chroma_collection = db.get_collection(collection_name)
        # 加载已有索引
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        print(f"✅ Loaded existing index from {persist_dir}")
    except NotFoundError:
        # 首次运行：构建新索引
        if nodes is None:
            raise ValueError("No existing index found and no nodes provided to build one.")
        print(f"🆕 Building new index and saving to {persist_dir}...")
        chroma_collection = db.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True
        )
        # 自动持久化（Chroma PersistentClient 会写入磁盘）
    
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