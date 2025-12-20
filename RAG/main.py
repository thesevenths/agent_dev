from loaders import load_pdf_with_tables
from chunkers import get_semantic_splitter
from retrievers import get_bm25_retriever, get_vector_retriever
from query_engine import build_query_engine
from pathlib import Path

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from config import LLAMA_CLOUD_API_KEY
print(f"main LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")

PDF_DIR = r"E:\\model\\report"

def main():
    print("1. Loading PDFs with LlamaParse...")
    documents = load_pdf_with_tables(PDF_DIR)
    print(f"Loaded {len(documents)} chunks (with table-aware parsing).")

    print("2. Chunking...")
    splitter = get_semantic_splitter()
    nodes = splitter.get_nodes_from_documents(documents)

    print("3. Building retrievers...")
    bm25_retriever = get_bm25_retriever(nodes, top_k=10)
    vector_retriever = get_vector_retriever(nodes, top_k=10)

    print("✅ RAG system ready! Ask questions like:")
    print('  - "2023年哪些公司毛利率超过50%？"')
    print('  - "对比宁德时代和比亚迪的净利润"')
    
    while True:
        query = input("\nYour question (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        
        # 关键：将原始 query 传入，用于动态元数据过滤
        query_engine = build_query_engine(bm25_retriever, vector_retriever, query)
        
        response = query_engine.query(query)
        print("\nAnswer:", response.response)
        print("\nSources:")
        for i, node in enumerate(response.source_nodes, 1):
            meta = node.node.metadata
            print(f"{i}. {meta.get('company', 'Unknown')} ({meta.get('fiscal_year', 'Unknown')}) - {meta.get('file_name', '')}")
            preview = node.node.text.replace("\n", " ")[:150]
            print(f"   Preview: {preview}...")

if __name__ == "__main__":
    main()