import os
import re
import json
from pathlib import Path
from typing import List, Dict
import markdown
from bs4 import BeautifulSoup
import jieba

# Document 兼容回退（保持不变）
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

                def get_text(self, *args, **kwargs) -> str:
                    return self.text

                def get_content(self, metadata_mode=None, *args, **kwargs) -> str:
                    return self.text

# ========== 配置 ==========
from config import MD_DIR, JSON_DIR
Path(JSON_DIR).mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
OVERLAP = 50

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def count_tokens(text: str) -> int:
    return len(jieba.lcut(text))

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    words = jieba.lcut(text)
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunk = ''.join(words[i:j])
        chunks.append(chunk)
        i = j - overlap
        if j == len(words):
            break
    return chunks

def extract_tables_from_md(md_content: str) -> List[List[List[str]]]:
    """
    提取 HTML 表格，返回 list of table，其中 table 是 list of rows，row 是 list of cells
    不依赖 pandas 列对齐，彻底避免列数不匹配报错
    """
    tables = []
    html = markdown.markdown(md_content, extensions=['tables'])
    soup = BeautifulSoup(html, 'html.parser')

    for table_tag in soup.find_all('table'):
        table_data = []
        for tr in table_tag.find_all('tr'):
            row = []
            for cell in tr.find_all(['td', 'th']):
                # 获取纯文本，处理可能的多行或嵌套
                text = cell.get_text(separator=' ', strip=True)
                row.append(text)
            if row:  # 忽略空行
                table_data.append(row)
        if table_data:
            tables.append(table_data)
            print(f"   提取到一个表格：{len(table_data)} 行，列数示例：{[len(r) for r in table_data]}")
    return tables

def table_to_json(table: List[List[str]]) -> Dict[str, str]:
    """将手动提取的 table 转为 {row_label:col_label: value}"""
    if not table or len(table) < 1:
        return {}

    table_json = {}
    headers = table[0]  # 第一行作为表头
    for row_idx in range(1, len(table)):
        row = table[row_idx]
        row_label = row[0].strip() if len(row) > 0 else f"未知行{row_idx}"
        for col_idx in range(1, len(headers)):
            if col_idx >= len(row):
                continue  # 补空
            col_label = headers[col_idx].strip()
            value = row[col_idx].strip()
            if value:
                key = f"{row_label}:{col_label}"
                table_json[key] = value
    return table_json

def _serialize_table(table_json: Dict[str, str]) -> str:
    return "\n".join([f"{k}\t{v}" for k, v in table_json.items()])

def process_md(md_path: Path) -> Dict:
    result = {
        "file_name": md_path.name,
        "text_chunks": [],
        "tables": {}
    }

    try:
        md_content = md_path.read_text(encoding="utf-8")
        filename_prefix = f"文件名: {md_path.name}\n"
        print(f"正在处理: {md_path.name} (长度 {len(md_content)} 字符)")

        # 提取表格（手动方式，更健壮）
        tables = extract_tables_from_md(md_content)

        # 去除表格内容（粗暴但有效）
        non_table_text = re.sub(r'<table>.*?</table>', '', md_content, flags=re.DOTALL | re.IGNORECASE)
        # 再去除可能的残余 HTML 标签
        non_table_text = re.sub(r'<[^>]+>', '', non_table_text)
        non_table_text = clean_text(non_table_text)

        # chunk 纯文本
        if non_table_text:
            chunks = chunk_text(non_table_text)
            print(f"   分出 {len(chunks)} 个文本 chunk")
            for idx, chunk in enumerate(chunks, 1):
                result["text_chunks"].append({
                    "chunk_id": idx,
                    "content": filename_prefix + chunk
                })

        # 处理表格
        for idx, table in enumerate(tables, 1):
            table_json = table_to_json(table)
            if not table_json:
                print(f"   表格 {idx} 为空，跳过")
                continue

            table_text = _serialize_table(table_json)
            tokens = count_tokens(table_text)
            key = f"{md_path.stem}_table_{idx}"

            if tokens > CHUNK_SIZE:
                print(f"   表格 {idx} 超大 ({tokens} tokens)，拆分成两部分")
                mid = len(table) // 2
                part1 = table[:mid]
                part2 = table[mid:]
                result["tables"][f"{key}_part1"] = table_to_json(part1)
                result["tables"][f"{key}_part2"] = table_to_json(part2)
            else:
                result["tables"][key] = table_json
                print(f"   表格 {idx} 完整保留 ({tokens} tokens)")

    except Exception as e:
        print(f"❌ 处理失败 {md_path.name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

    return result

def process_mds_to_json(md_dir: str = None, json_dir: str = JSON_DIR, force: bool = False) -> List[str]:
    md_dir = Path(md_dir or MD_DIR)
    json_dir = Path(json_dir or JSON_DIR)
    json_dir.mkdir(parents=True, exist_ok=True)

    md_files = list(md_dir.glob("*.md"))
    if not md_files:
        print(f"⚠️ 未找到 MD 文件 in {md_dir}")
        return []

    json_paths = []
    for md_file in md_files:
        out_file = json_dir / f"{md_file.stem}.json"

        if not force and out_file.exists() and out_file.stat().st_mtime >= md_file.stat().st_mtime:
            print(f"⏭ 跳过（已缓存）：{out_file.name}")
            json_paths.append(str(out_file))
            continue

        print(f"📄 处理: {md_file.name}")
        data = process_md(md_file)
        if data is None:
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        json_paths.append(str(out_file))
        print(f"✅ 保存 JSON: {out_file.name}")

    return json_paths

def load_items_from_json(json_dir: str = JSON_DIR) -> List[Document]:
    docs = []
    json_dir = Path(json_dir or JSON_DIR)
    for p in json_dir.glob("*.json"):
        try:
            obj = json.load(open(p, encoding="utf-8"))
        except:
            continue
        source = obj.get("file_name", p.stem)
        prefix = f"文件名: {source}\n"

        for t in obj.get("text_chunks", []):
            docs.append(Document(text=t.get("content", ""), metadata={"source": source, "is_table": False}))

        for tbl_key, tbl in obj.get("tables", {}).items():
            text = prefix + _serialize_table(tbl)
            docs.append(Document(text=text, metadata={"source": source, "table_id": tbl_key, "is_table": True}))

    print(f"从 JSON 加载了 {len(docs)} 个 Document（含表格）")
    return docs

def main():
    json_paths = process_mds_to_json(MD_DIR, JSON_DIR, force=True)  # 建议第一次 force=True
    print(f" -> {len(json_paths)} json files ready")

if __name__ == "__main__":
    main()