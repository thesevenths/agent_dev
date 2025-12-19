from llama_index.core.node_parser import SimpleNodeParser

def get_semantic_splitter():
    """
    LlamaParse 输出已接近语义块（每表/每段为一块），
    这里仅做轻量分块，避免切碎表格。
    """
    return SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=20,
        include_metadata=True
    )