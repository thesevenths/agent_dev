from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.llama_parse import LlamaParse
import json
import hashlib
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from config import LLAMA_CLOUD_API_KEY
print(f"out LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")

from pathlib import Path
# 缓存目录（可自定义）
CACHE_DIR = Path("E:\\model\\RAG\\llama_parse")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def chinese_to_arabic_year(chinese: str) -> str:
    """将“二〇二三年”转为“2023”"""
    chinese = chinese.replace("年", "")
    digit_map = {"〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
                 "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
    arabic = "".join(digit_map.get(char, "") for char in chinese)
    if len(arabic) == 4 and arabic.startswith("20"):
        return arabic
    return "Unknown"

def extract_metadata_from_text(text: str):
    """从文档前500字符中自动提取公司名和年份"""
    preview = text[:500]
    
    # 1. 提取年份
    year = "Unknown"
    year_match = re.search(r"(202[0-9])", preview)
    if year_match:
        year = year_match.group(1)
    else:
        chinese_year_match = re.search(r"[二三四五六七八九〇一二三四五六七八九]{4,6}年", preview)
        if chinese_year_match:
            year = chinese_to_arabic_year(chinese_year_match.group(0))
    
    # 2. 提取公司名称（更鲁棒）
    company = "Unknown"
    company_match = re.search(r"([\u4e00-\u9fa5]{4,20}?(?:公司|集团|股份有限公司|有限公司|股份))", preview)
    if company_match:
        company = company_match.group(1)
        # 精确去除常见后缀
        company = re.sub(r'(股份)?有限公司$', '', company)
        company = re.sub(r'(股份)?公司$', '', company)
        company = re.sub(r'(有限)?责任公司$', '', company)  # 覆盖“有限责任公司”
        company = company.rstrip('股份集团')  # 去除尾部冗余词
        company = company or "Unknown"  # 防空

    return {"company": company.strip(), "fiscal_year": year}

def has_markdown_table(text: str) -> bool:
    """判断 Markdown 文本是否包含表格（更准确）"""
    # 方法1：快速检查
    if "||" not in text:
        return False
    # 方法2：正则确认（可选，性能略低）
    table_pattern = r"\|\s*[^|\n]*\|\s*\n\|[-:\s|]+\|"
    return bool(re.search(table_pattern, text[:2000]))  # 只查前2000字符加速

def get_cache_key(file_path: str) -> str:
    """基于文件路径和修改时间生成缓存键"""
    stat = os.stat(file_path)
    key_str = f"{file_path}:{stat.st_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_or_parse_pdf(file_path: str, parser: LlamaParse) -> Document:
    """带缓存的单个 PDF 解析"""
    cache_key = get_cache_key(file_path)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        # 从缓存加载
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Document(text=data["text"], metadata=data["metadata"])

    # 调用 LlamaParse
    print(f"Parsing PDF (not cached): {file_path}")
    docs = parser.load_data(file_path)
    if not docs:
        doc = Document(text="", metadata={"file_path": file_path})
    else:
        doc = docs[0]  # LlamaParse 通常返回单个 Document

    # 保存缓存
    cache_data = {
        "text": doc.text,
        "metadata": doc.metadata
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

    return doc

def load_pdf_with_tables(pdf_dir: str):
    print(f"in LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        language="ch_sim",
        ignore_errors=True,
    )

    pdf_files = [str(p) for p in Path(pdf_dir).glob("*.pdf")]
    documents = []

    for pdf_file in pdf_files:
        doc = load_or_parse_pdf(pdf_file, parser)
        fname = os.path.basename(doc.metadata.get("file_path", pdf_file))

        # 提取元数据
        meta = extract_metadata_from_text(doc.text)
        meta["file_name"] = fname
        meta["has_table"] = has_markdown_table(doc.text)

        doc.metadata.update(meta)
        documents.append(doc)

    return documents