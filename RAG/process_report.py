import os
import re
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import markdown  # 新增：用于解析MD
import io  # 用于pandas读取MD表格
import jieba  # 用于token计数和分词

# 尝试导入 Document（兼容不同 llama-index 版本），若不可用则提供简单回退
try:
    from llama_index import Document
except Exception:
    try:
        from llama_index.schema import Document
    except Exception:
        try:
            from llama_index.node import Document
        except Exception:
            import uuid, hashlib, json
            from dataclasses import dataclass, field
            from typing import Any, Dict

            @dataclass
            class Document:
                text: str
                metadata: dict = field(default_factory=dict)
                id_: str = None
                hash: str = None
                doc_id: str = None

                def __post_init__(self):
                    if not self.id_:
                        self.id_ = (
                            self.metadata.get("item_id")
                            or self.metadata.get("table_id")
                            or self.metadata.get("source")
                            or self.doc_id
                            or str(uuid.uuid4())
                        )
                    if not self.doc_id:
                        self.doc_id = self.id_
                    if not self.hash:
                        meta_str = json.dumps(self.metadata or {}, sort_keys=True, ensure_ascii=False)
                        h = hashlib.md5()
                        h.update((self.text or "").encode("utf-8"))
                        h.update(meta_str.encode("utf-8"))
                        self.hash = h.hexdigest()

                def model_dump(self, mode: str = "json") -> Dict[str, Any]:
                    out = dict(self.metadata) if isinstance(self.metadata, dict) else {}
                    out.update({"text": self.text, "id_": self.id_, "doc_id": self.doc_id})
                    return out

                def get_metadata_str(self, mode=None, **kwargs) -> str:
                    return json.dumps(self.metadata or {}, ensure_ascii=False, sort_keys=True)

                def get_text(self, *args, **kwargs) -> str:
                    return self.text

                def get_content(self, metadata_mode=None, *args, **kwargs) -> str:
                    return self.text

                def class_name(self) -> str:
                    return self.__class__.__name__

                def as_related_node_info(self) -> Dict[str, Any]:
                    """
                    返回一个用于构建关系的简单结构。
                    llama-index 期望有 as_related_node_info()，这里返回常见字段的字典。
                    """
                    return {
                        "doc_id": self.id_,
                        "node_id": self.id_,
                        "extra_info": dict(self.metadata or {})
                    }

                def __repr__(self):
                    return f"<FallbackDocument id_={self.id_} source={self.metadata.get('source')}>"

# ========== 配置 ==========
from config import MD_DIR, JSON_DIR  # 从config导入
Path(JSON_DIR).mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500  # token数（用jieba词数近似）
OVERLAP = 50

def clean_text(text: str):
    """清理换行等杂字符"""
    return re.sub(r'\s+', ' ', text).strip()

def count_tokens(text: str) -> int:
    """用jieba分词计数作为token近似"""
    return len(jieba.lcut(text))

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """将文本chunk成指定大小，overlap"""
    words = jieba.lcut(text)
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if end >= len(words):
            break
    return chunks

def extract_tables_from_md(md_content: str) -> List[pd.DataFrame]:
    """从MD提取表格块，返回list of DataFrame"""
    tables = []
    # 匹配MD表格：至少有header和divider
    table_pattern = r'(\|.*?\n\|[-:\s\|]+\n(?:\|.*?\n)+)'
    for match in re.finditer(table_pattern, md_content, re.MULTILINE):
        table_str = match.group(1)
        try:
            df = pd.read_csv(io.StringIO(table_str), sep='|', engine='python').dropna(how='all', axis=1)
            df.columns = df.columns.str.strip()
            tables.append(df)
        except:
            pass  # 忽略解析失败的表格
    return tables

def process_md(md_path: Path) -> Dict:
    """
    解析单个MD，提取文本和表格，返回JSON结构（dict）
    """
    result = {
        "file_name": md_path.name,
        "text_chunks": [],  # list of {"content": chunk_text} （已chunk）
        "tables": {}       # key -> {row:col: value, ...}
    }

    try:
        md_content = md_path.read_text(encoding="utf-8")
        filename_prefix = f"文件名: {md_path.name}\n"

        # 转换为HTML以辅助提取纯文本（去除表格）
        html = markdown.markdown(md_content)
        # 但实际我们用正则去除表格块，剩余为文本
        non_table_text = re.sub(r'(\|.*?\n\|[-:\s\|]+\n(?:\|.*?\n)+)', '', md_content, flags=re.MULTILINE)
        non_table_text = clean_text(non_table_text)

        # chunk非表格文本
        if non_table_text:
            chunks = chunk_text(non_table_text)
            for idx, chunk in enumerate(chunks):
                chunk_with_prefix = filename_prefix + chunk
                result["text_chunks"].append({
                    "chunk_id": idx + 1,
                    "content": chunk_with_prefix
                })

        # 提取表格
        tables = extract_tables_from_md(md_content)
        for idx, df in enumerate(tables, 1):
            table_json = table_dataframe_to_json(df)
            table_text = _serialize_table(table_json)
            table_tokens = count_tokens(table_text)
            key = f"{md_path.stem}_table_{idx}"

            if table_tokens > CHUNK_SIZE:
                # 分成两个chunk：大致二分行
                mid = len(df) // 2
                df1 = df.iloc[:mid]
                df2 = df.iloc[mid:]
                table_json1 = table_dataframe_to_json(df1)
                table_json2 = table_dataframe_to_json(df2)
                result["tables"][f"{key}_part1"] = table_json1
                result["tables"][f"{key}_part2"] = table_json2
            else:
                result["tables"][key] = table_json

    except Exception as e:
        print(f"❌ 处理失败 {md_path.name}: {e}")
        return None

    return result

def table_dataframe_to_json(df: pd.DataFrame) -> Dict[str, str]:
    """
    把 DataFrame 转为 { "row:col": value, ... } 结构
    """
    table_json: Dict[str, str] = {}
    if df.shape[0] < 2:
        return table_json

    headers = df.iloc[0].fillna("").tolist()
    body = df.iloc[1:].fillna("").values.tolist()

    for row in body:
        row_label = str(row[0]).strip()
        if not row_label:
            continue
        for col_idx in range(1, len(headers)):
            col_label = str(headers[col_idx]).strip()
            raw_val = str(row[col_idx]).strip()
            # 过滤掉无用字符
            if raw_val:
                key = f"{row_label}:{col_label}"
                table_json[key] = raw_val

    return table_json

def _serialize_table(table) -> str:
    """
    将表格对象序列化为单字符串（不可拆分单元）
    """
    if table is None:
        return ""
    if isinstance(table, dict):
        lines = []
        for k, v in table.items():
            lines.append(f"{k}\t{v}")
        return "\n".join(lines)
    try:
        rows = []
        for r in table:
            rows.append("\t".join([str(c) for c in r]))
        return "\n".join(rows)
    except Exception:
        return str(table)


def process_mds_to_json(md_dir: str = None, json_dir: str = JSON_DIR, force: bool = False) -> List[str]:
    """
    遍历 md_dir 下的所有 md，逐个调用 process_md，把结果保存到 json_dir（按文件名 .json）。
    支持基于修改时间跳过已存在的 json（除非 force=True）。
    返回已生成（或存在）的 json 文件路径列表。
    """
    md_dir = Path(md_dir or MD_DIR)
    json_dir = Path(json_dir or JSON_DIR)
    json_dir.mkdir(parents=True, exist_ok=True)

    if not md_dir.exists():
        print(f"❌ MD 目录不存在: {md_dir}")
        return []

    md_files = list(md_dir.glob("*.md"))
    if not md_files:
        print(f"⚠️ {md_dir} 下没有 MD 文件")
        return []

    json_paths: List[str] = []
    for md_file in md_files:
        out_file = json_dir / f"{md_file.stem}.json"
        need_write = force
        try:
            if out_file.exists():
                # 如果 json 比 md 新且非 force，则跳过
                if not force and out_file.stat().st_mtime >= md_file.stat().st_mtime:
                    json_paths.append(str(out_file))
                    print(f"⏭ 跳过（已缓存）：{out_file.name}")
                    continue
            need_write = True
        except Exception:
            need_write = True

        print(f"📄 处理: {md_file.name}")
        json_data = process_md(md_file)
        if json_data is None:
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        json_paths.append(str(out_file))
        print(f"✅ 已保存 JSON: {out_file.name}")

    return json_paths

def load_items_from_json(json_dir: str = JSON_DIR) -> List[Document]:
    """
    从 json_dir 读取所有 json 文件，返回 llama_index.Document 列表。
    每个 text chunk -> 一个 Document；每个 table -> 一个 Document（is_table metadata）
    """
    docs: List[Document] = []
    json_dir = Path(json_dir or JSON_DIR)
    for p in json_dir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = obj.get("file_name", p.stem)
        # text chunks
        for t in obj.get("text_chunks", []):
            meta = {"source": source, "is_table": False}
            docs.append(Document(text=t.get("content", ""), metadata=meta))
        # tables
        for tbl_key, tbl in obj.get("tables", {}).items():
            meta = {"source": source, "table_id": tbl_key, "is_table": True}
            filename_prefix = f"文件名: {source}\n"
            txt = filename_prefix + _serialize_table(tbl)
            docs.append(Document(text=txt, metadata=meta))
    return docs

def main():
    json_paths = process_mds_to_json(MD_DIR, json_dir=JSON_DIR, force=False)
    print(f" -> {len(json_paths)} json files ready in {JSON_DIR}")

if __name__ == "__main__":
    main()