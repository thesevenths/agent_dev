import os
import re
import pdfplumber
import camelot
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict

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
PDF_DIR = r"E:\model\RAG\report"
OUTPUT_DIR = r"E:\model\RAG\report_json_camelot"
JSON_DIR = r"E:\model\RAG\json_reports"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(JSON_DIR).mkdir(parents=True, exist_ok=True)


def clean_text(text: str):
    """清理换行等杂字符"""
    return text.replace("\r", " ").replace("\n", " ").strip()


def table_dataframe_to_json(df: pd.DataFrame) -> Dict[str, str]:
    """
    把 camelot 的 DataFrame 转为 { "row:col": value, ... } 结构
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


def extract_tables_with_camelot(pdf_path: Path, page: int) -> List[Dict[str, str]]:
    """
    用 camelot 提取一个页面的所有表格，返回 list of dict (row:col -> value)
    """
    all_tables: List[Dict[str, str]] = []

    # 优先用 lattice（基于线条）
    try:
        lattice_tables = camelot.read_pdf(str(pdf_path), pages=str(page), flavor="lattice")
        for t in lattice_tables:
            df = t.df
            j = table_dataframe_to_json(df)
            if j:
                all_tables.append(j)
    except Exception as e:
        print(f"⚠ lattice fail page {page}: {e}")

    # 再尝试 stream（基于文本流）
    try:
        stream_tables = camelot.read_pdf(str(pdf_path), pages=str(page), flavor="stream")
        for t in stream_tables:
            df = t.df
            j = table_dataframe_to_json(df)
            if j:
                all_tables.append(j)
    except Exception as e:
        print(f"⚠ stream fail page {page}: {e}")

    return all_tables


def process_pdf(pdf_path: Path) -> Dict:
    """
    解析单个 PDF，提取文本和所有表格，返回 JSON 结构（dict）
    """
    result = {
        "file_name": pdf_path.name,
        "text": [],     # list of {"page": page_num, "content": text}
        "tables": {}    # key -> {row:col: value, ...}
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # 提取文字
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                page_text = clean_text(page_text)
                if page_text:
                    result["text"].append({
                        "page": page_num,
                        "content": page_text
                    })

                # 提取表格（camelot）
                tables = extract_tables_with_camelot(pdf_path, page_num)
                for idx, table in enumerate(tables, 1):
                    key = f"{pdf_path.stem}_table_{idx}_page_{page_num}"
                    result["tables"][key] = table

    except Exception as e:
        print(f"❌ 处理失败 {pdf_path.name}: {e}")
        return None

    return result


def _serialize_table(table) -> str:
    """
    将表格对象（dict row:col -> value 或 list-of-rows）序列化为单字符串（不可拆分单元）
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


def process_pdfs_to_json(pdf_dir: str = None, json_dir: str = JSON_DIR, force: bool = False) -> List[str]:
    """
    遍历 pdf_dir 下的所有 pdf，逐个调用 process_pdf，把结果保存到 json_dir（按文件名 .json）。
    支持基于修改时间跳过已存在的 json（除非 force=True）。
    返回已生成（或存在）的 json 文件路径列表。
    """
    pdf_dir = Path(pdf_dir or PDF_DIR)
    json_dir = Path(json_dir or JSON_DIR)
    json_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        print(f"❌ PDF 目录不存在: {pdf_dir}")
        return []

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ {pdf_dir} 下没有 PDF 文件")
        return []

    json_paths: List[str] = []
    for pdf_file in pdf_files:
        out_file = json_dir / f"{pdf_file.stem}.json"
        need_write = force
        try:
            if out_file.exists():
                # 如果 json 比 pdf 新且非 force，则跳过
                if not force and out_file.stat().st_mtime >= pdf_file.stat().st_mtime:
                    json_paths.append(str(out_file))
                    print(f"⏭ 跳过（已缓存）：{out_file.name}")
                    continue
            need_write = True
        except Exception:
            need_write = True

        print(f"📄 处理: {pdf_file.name}")
        json_data = process_pdf(pdf_file)
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
    每个 text item -> 一个 Document；每个 table -> 一个 Document（is_table metadata）
    """
    docs: List[Document] = []
    json_dir = Path(json_dir or JSON_DIR)
    for p in json_dir.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        source = obj.get("file_name", p.stem)
        # texts
        for t in obj.get("text", []):
            meta = {"source": source, "page": t.get("page"), "is_table": False}
            docs.append(Document(text=t.get("content", ""), metadata=meta))
        # tables
        for tbl_key, tbl in obj.get("tables", {}).items():
            meta = {"source": source, "table_id": tbl_key, "is_table": True}
            txt = _serialize_table(tbl)
            docs.append(Document(text=txt, metadata=meta))
    return docs


def main():
    json_paths = process_pdfs_to_json(PDF_DIR, json_dir=JSON_DIR, force=False)
    print(f" -> {len(json_paths)} json files ready in {JSON_DIR}")


if __name__ == "__main__":
    main()
