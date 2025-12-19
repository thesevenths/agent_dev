from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.embeddings import resolve_embed_model
from config import EMBEDDING_MODEL, DASHSCOPE_API_KEY
from llama_index.embeddings.dashscope import DashScopeEmbedding

def get_semantic_splitter():
    # 使用 DashScope 的 embedding 模型进行语义分块
    embed_model = DashScopeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY
    )
    # SemanticSplitter 会基于句子嵌入的相似度动态切分
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,  # 相邻句子数量用于判断语义边界
        breakpoint_percentile_threshold=95  # 相似度低于95%分位数处切分
    )
    return splitter