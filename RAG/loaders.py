from llama_index.core import SimpleDirectoryReader
from llama_index.readers.llama_parse import LlamaParse

import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from config import LLAMA_CLOUD_API_KEY
print(f"out LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")

def extract_metadata_from_text(text: str):
    """从文档前500字符中自动提取公司名和年份"""
    preview = text[:500]
    
    # 1. 提取年份（支持阿拉伯数字和中文数字）
    year = "Unknown"
    # 匹配 2020-2029
    year_match = re.search(r"(202[0-9])", preview)
    if year_match:
        year = year_match.group(1)
    else:
        # 匹配“二〇二三年” -> 转为 2023
        chinese_year_match = re.search(r"[二三四五六七八九〇一二三四五六七八九]{4,6}年", preview)
        if chinese_year_match:
            chinese_year = chinese_year_match.group(0)
            year = chinese_to_arabic_year(chinese_year)
    
    # 2. 提取公司名称
    company = "Unknown"
    # 尝试匹配“XXXX公司”、“XXXX股份有限公司”等模式
    # 公司名通常在标题附近，且长度 >= 4 个中文字符
    company_match = re.search(r"([\u4e00-\u9fa5]{4,20}?(?:公司|集团|股份))", preview)
    if company_match:
        company = company_match.group(1)
        # 清理后缀（可选）
        if company.endswith("有限"):
            company = company[:-2]
    
    return {"company": company.strip(), "fiscal_year": year}

def chinese_to_arabic_year(chinese: str) -> str:
    """将“二〇二三年”转为“2023”"""
    chinese = chinese.replace("年", "")
    digit_map = {"〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
                 "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
    arabic = ""
    for char in chinese:
        if char in digit_map:
            arabic += digit_map[char]
    # 确保是4位年份
    if len(arabic) == 4 and arabic.startswith("20"):
        return arabic
    return "Unknown"

def load_pdf_with_tables(pdf_dir: str):
    from config import LLAMA_CLOUD_API_KEY
    print(f"in LLAMA_CLOUD_API_KEY: {LLAMA_CLOUD_API_KEY}")
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",
        language="ch_sim",
        ignore_errors=True,
    )

    reader = SimpleDirectoryReader(
        input_dir=pdf_dir,
        required_exts=[".pdf"],
        file_extractor={".pdf": parser},
    )

    documents = reader.load_data()

    for doc in documents:
        fname = os.path.basename(doc.metadata.get("file_path", ""))
        
        # 从文档内容提取元数据（关键改进！）
        meta = extract_metadata_from_text(doc.text)
        meta["file_name"] = fname
        
        # 标记是否含表格
        meta["has_table"] = "|" in doc.text[:200]  # 简易判断
        
        doc.metadata.update(meta)

    return documents