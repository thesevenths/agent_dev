import jieba
import pickle
from pathlib import Path

NODES_CACHE = "E:\\model\\RAG\\nodes.pkl"

# 测试 query
test_queries = [
    "对比高伟达、京北方、宇信科技三家公司的2025年报告，分析其核心财务指标，告诉我哪家公司最具投资价值"
]

# 查看 jieba 分词结果
print("=" * 60)
print("🔍 Jieba 分词结果:")
print("=" * 60)
for q in test_queries:
    tokens = jieba.lcut(q)
    print(f"Query: {q}")
    print(f"Tokens: {tokens}")
    print()

# 加载缓存的 nodes，看看文档中实际有哪些词
print("=" * 60)
print("📄 文档样本分词结果:")
print("=" * 60)
with open(NODES_CACHE, "rb") as f:
    documents = pickle.load(f)

print(f"Total documents: {len(documents)}")
for i, doc in enumerate(documents[:5]):  # 只看前 5 个
    text = getattr(doc, "text", "")[:200]  # 只看前 200 字
    meta = getattr(doc, "metadata", {})
    tokens = jieba.lcut(text)
    print(f"\nDoc {i}:")
    print(f"  Source: {meta.get('source')}, is_table: {meta.get('is_table')}")
    print(f"  Text preview: {text}")
    print(f"  Tokens: {tokens[:20]}")  # 只显示前 20 个 token

# 手动测试 BM25 检索
print("\n" + "=" * 60)
print("🔎 BM25 手动检索测试:")
print("=" * 60)
from retrievers import get_bm25_retriever

bm25_ret = get_bm25_retriever(documents, top_k=15)
for q in test_queries:
    print(f"\nQuery: {q}")
    results = bm25_ret.retrieve(q)
    print(f"Found {len(results)} results:")
    for j, node in enumerate(results, 1):
        text = getattr(node, "text", "")[:150]
        meta = getattr(node, "metadata", {})
        print(f"  {j}. [{meta.get('source')}] {text}...")