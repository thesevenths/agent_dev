from loaders import load_pdf_with_tables
from chunkers import get_semantic_splitter
from retrievers import get_bm25_retriever, get_vector_retriever
from query_engine import build_query_engine
from pathlib import Path
import pickle

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from config import LLAMA_CLOUD_API_KEY
from config import DASHSCOPE_API_KEY
# print(f"main LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")

PDF_DIR = r"E:\\model\\RAG\\report"
CHROMA_PATH = "E:\\model\\RAG\\chroma_db"
NODES_CACHE = "E:\\model\\RAG\\nodes.pkl"

def main():
    # 检查是否已有持久化向量库和 nodes 缓存
    has_chroma = os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH)
    has_nodes = os.path.exists(NODES_CACHE)

    if has_chroma and has_nodes:
        print("🔍 Loading from persistent storage (Chroma + nodes.pkl)...")
        documents = None
        nodes = None
        # 先加载 nodes（用于 BM25）
        with open(NODES_CACHE, "rb") as f:
            nodes = pickle.load(f)
    else:
        print("1. Loading PDFs with LlamaParse...")
        documents = load_pdf_with_tables(PDF_DIR)
        print(f"Loaded {len(documents)} chunks (with table-aware parsing).")

        print("2. Chunking...")
        splitter = get_semantic_splitter()
        nodes = splitter.get_nodes_from_documents(documents)

        # 保存 nodes.pkl（用于后续跳过 PDF 解析）
        with open(NODES_CACHE, "wb") as f:
            pickle.dump(nodes, f)
        print(f"✅ Cached nodes to {NODES_CACHE}")

    print("3. Building retrievers...")
    # 向量检索器：自动处理 Chroma 持久化（在 retrievers.py 中实现）
    vector_retriever = get_vector_retriever(nodes, top_k=15)

    # BM25 检索器：使用已加载或刚生成的 nodes
    bm25_retriever = get_bm25_retriever(nodes, top_k=15)

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
            meta = node.node.metadata
            print(f"{i}. {meta.get('company', 'Unknown')} ({meta.get('fiscal_year', 'Unknown')})")
            preview = node.node.text.replace("\n", " ")[:150]
            print(f"   Preview: {preview}...")

if __name__ == "__main__":
    main()