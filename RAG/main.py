import os
from process_report import process_mds_to_json, load_items_from_json  # 修改为mds
from retrievers import get_bm25_retriever, get_vector_retriever
from query_engine import build_query_engine
from pathlib import Path
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from config import LLAMA_CLOUD_API_KEY
from config import DASHSCOPE_API_KEY
from config import MD_DIR, JSON_DIR  # 新增从config导入

# 尝试导入官方 Document；若失败则用 process_report 的回退版本
try:
    from llama_index.core import Document as OfficalDocument
except Exception:
    OfficalDocument = None

CHROMA_PATH = "E:\\model\\RAG\\chroma_db"
NODES_CACHE = "E:\\model\\RAG\\nodes.pkl"

def _convert_documents(docs):
    """
    将 process_report.Document 或其它 Document 转换为官方 llama-index Document（若可用）。
    若官方 Document 不可用则保持原样。
    """
    if OfficalDocument is None:
        return docs
    
    converted = []
    for d in docs:
        # 若已是官方 Document，跳过
        if isinstance(d, OfficalDocument):
            converted.append(d)
        else:
            # 转换：从回退 Document 提取 text 和 metadata
            txt = getattr(d, "text", "") or ""
            meta = getattr(d, "metadata", {}) or {}
            converted.append(OfficalDocument(text=txt, metadata=meta))
    return converted

def main():
    # 1) 先把 MD 转为 json（有缓存则跳过）
    print("1. Converting MDs to JSON (cached)...")
    json_paths = process_mds_to_json(MD_DIR, json_dir=JSON_DIR, force=False)
    print(f" -> {len(json_paths)} json files ready in {JSON_DIR}")

    # 2) 从 json 加载 items（每个 text chunk 与每个 table 都变成一个 Document）
    has_nodes = os.path.exists(NODES_CACHE)
    if has_nodes:
        print("🔍 Loading documents cache from nodes.pkl...")
        with open(NODES_CACHE, "rb") as f:
            documents = pickle.load(f)
    else:
        print("2. Loading items from JSON into Documents...")
        documents = load_items_from_json(JSON_DIR)
        with open(NODES_CACHE, "wb") as f:
            pickle.dump(documents, f)
        print(f"✅ Cached documents to {NODES_CACHE} (total {len(documents)})")

    # 转换为官方 Document（如果可用）
    documents = _convert_documents(documents)

    print("3. Building retrievers...")
    # 向量检索器：自动处理 Chroma 持久化（在 retrievers.py 中实现）
    vector_retriever = get_vector_retriever(documents=documents, top_k=25)

    # BM25 检索器：用 documents（每个 item 一个 Document）
    bm25_retriever = get_bm25_retriever(documents=documents, top_k=30)

    print("✅ RAG system ready!")
    while True:
        query = input("\nYour question (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        query_engine = build_query_engine(bm25_retriever, vector_retriever, query)
        response = query_engine.query(query)
        print("\nAnswer:", response.response)
        print("\nSources:")
        for i, node in enumerate(response.source_nodes, 1):
            meta = getattr(node.node, "metadata", {}) if hasattr(node, "node") else getattr(node, "metadata", {})
            print(f"{i}. {meta.get('source', 'Unknown')} (table={meta.get('is_table', False)})")
            preview = (getattr(node.node, "text", "") if hasattr(node, "node") else getattr(node, "text", ""))[:150].replace("\n", " ")
            print(f"   Preview: {preview}...")

if __name__ == "__main__":
    main()