from loaders import load_pdf_with_tables
from chunkers import get_semantic_splitter
from retrievers import get_bm25_retriever, get_vector_retriever
from query_engine import build_query_engine

PDF_DIR = r"E:\\model\\report"

def main():
    print("1. Loading PDFs...")
    documents = load_pdf_with_tables(PDF_DIR)
    print(f"Loaded {len(documents)} document pages.")

    print("2. Semantic chunking...")
    splitter = get_semantic_splitter()
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Generated {len(nodes)} semantic chunks.")

    print("3. Building retrievers...")
    bm25_retriever = get_bm25_retriever(nodes, top_k=5)
    vector_retriever = get_vector_retriever(nodes, top_k=5)

    print("4. Building query engine...")
    query_engine = build_query_engine(bm25_retriever, vector_retriever)

    print("✅ RAG system ready!")
    while True:
        query = input("\nEnter your question (or 'quit'): ")
        if query.strip().lower() == 'quit':
            break
        response = query_engine.query(query)
        print("\nAnswer:", response.response)
        print("\nSources:")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"{i}. {node.node.metadata.get('file_name', 'Unknown')} (score: {node.score:.3f})")
            print(f"   Preview: {node.node.text[:200]}...")

if __name__ == "__main__":
    main()