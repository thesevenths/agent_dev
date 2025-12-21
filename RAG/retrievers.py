from typing import List
import jieba
import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
import uuid

# 尝试兼容不同版本的 llama-index 导入
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception as e:
    raise ImportError("需要安装兼容的 llama-index 包，或检查版本。详细错误: " + str(e))

# 向量索引相关
try:
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import StorageContext
except Exception:
    # 如果有版本差异，这里保留导入失败信息供调试
    raise

# DashScope embedding（若不需要可改为其他 embedding）
from config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
try:
    from llama_index.embeddings.dashscope import DashScopeEmbedding
except Exception:
    DashScopeEmbedding = None  # 回退：如果不可用，后续构建向量索引时会报更明确的错误

# 兼容 Document 的导入
try:
    from llama_index import Document
except Exception:
    try:
        from llama_index.schema import Document
    except Exception:
        from dataclasses import dataclass
        @dataclass
        class Document:
            text: str
            metadata: dict = None

# 兼容 TextNode 的导入
try:
    from llama_index.core.schema import TextNode
except Exception:
    try:
        from llama_index.schema import TextNode
    except Exception:
        # 如果实在没有，用最简兼容实现，但必须继承 BaseNode（如果可用）
        try:
            from llama_index.core.schema import BaseNode
        except Exception:
            from llama_index.schema import BaseNode  # 再尝试一次

        class TextNode(BaseNode):
            def __init__(self, text: str, metadata: dict = None, id_: str = None):
                from uuid import uuid4
                self._text = text
                self._metadata = metadata or {}
                self._id = id_ or str(uuid4())
                super().__init__(text=text, metadata=metadata, id_=self._id)

            @property
            def text(self) -> str:
                return self._text

            @property
            def metadata(self) -> dict:
                return self._metadata

            @property
            def node_id(self) -> str:
                return self._id

            def get_content(self, *args, **kwargs) -> str:
                return self.text

            def get_text(self) -> str:
                return self.text

            def get_metadata_str(self) -> str:
                return str(self.metadata)

# 兼容 SimpleNodeParser 的导入；若不存在则使用简单回退实现
try:
    from llama_index.node_parser import SimpleNodeParser
except Exception:
    import uuid
    class SimpleNodeParser:
        def get_nodes_from_documents(self, documents):
            nodes = []
            # 在 get_nodes_from_documents 中：
            for i, d in enumerate(documents):
                txt = getattr(d, "text", "") or ""
                meta = getattr(d, "metadata", {}) or {}
                source = meta.get("source", "unknown")
                
                # 优先使用文档自带 ID，否则用 source + UUID
                doc_id = meta.get("item_id") or meta.get("table_id") or meta.get("source")
                if doc_id:
                    nid = f"{doc_id}_{str(uuid.uuid4())[:8]}"
                else:
                    nid = str(uuid.uuid4())
                
                node = TextNode(text=txt, metadata=meta, id_=nid)
                nodes.append(node)
            return nodes

CHROMA_PATH = "E:\\model\\RAG\\chroma_db"

def get_bm25_retriever(documents: list, top_k=5, filters=None):
    """
    使用 SimpleNodeParser 将 Document 列表转换为节点，再构建 BM25Retriever。
    documents: list[llama_index.Document]
    """
    if not documents:
        raise ValueError("documents required for BM25 retriever")

    def tokenize(text: str):
        return jieba.lcut(text)

    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    return BM25Retriever.from_defaults(
        nodes=nodes,
        tokenizer=tokenize,
        similarity_top_k=top_k,
        filters=filters
    )

def get_vector_retriever(documents: list=None, top_k=5, filters=None, persist_dir=CHROMA_PATH):
    """
    如果 persist_dir 存在且含有 collection，则加载已有索引；
    否则用 documents 构建新索引并持久化。
    documents: list[llama_index.Document]
    """
    if DashScopeEmbedding is None:
        raise ImportError("DashScopeEmbedding 不可用，请安装相应 llama-index embeddings 或修改为其它 embedding 实现。")

    embed_model = DashScopeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        batch_size=1  # 改为 1，给 API 更多余量
    )

    # 初始化 Chroma 客户端（持久化）
    db = chromadb.PersistentClient(path=persist_dir)
    collection_name = "annual_reports"

    try:
        chroma_collection = db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        print(f"✅ Loaded existing index from {persist_dir}")
    except NotFoundError:
        if documents is None:
            raise ValueError("No existing index found and no documents provided to build one.")
        print(f"🆕 Building new index and saving to {persist_dir}...")
        chroma_collection = db.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        # 分批，每批最多 10 个 node
        batch_size = 10
        index = None
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            if i == 0:
                index = VectorStoreIndex(
                    nodes=batch,
                    embed_model=embed_model,
                    storage_context=storage_context,
                    show_progress=True
                )
            else:
                index.insert_nodes(batch)
    return index.as_retriever(similarity_top_k=top_k, filters=filters)

def filter_nodes_by_metadata(nodes, filters_dict):
    """简易元数据过滤（用于 BM25）"""
    if not filters_dict:
        return nodes
    filtered = []
    for node in nodes:
        meta = getattr(node, "metadata", {}) or {}
        match = True
        for key, value in filters_dict.items():
            if meta.get(key) != value:
                match = False
                break
        if match:
            filtered.append(node)
    return filtered