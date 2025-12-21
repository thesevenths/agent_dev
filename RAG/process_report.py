import os
import re
import pdfplumber
import camelot
import pandas as pd
import json
from pathlib import Path

# ========== 配置 ==========
PDF_DIR = r"E:\model\RAG\report"
OUTPUT_DIR = r"E:\model\RAG\report_json_camelot"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def clean_text(text: str):
    """清理换行等杂字符"""
    return text.replace("\n", " ").strip()


def table_dataframe_to_json(df: pd.DataFrame):
    """
    把 camelot 的 DataFrame 转为 { "row:col": value, ... } 结构
    """
    table_json = {}
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


def extract_tables_with_camelot(pdf_path: Path, page):
    """
    用 camelot 提取一个页面的所有表格
    """
    all_tables = []

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


def process_pdf(pdf_path: Path):
    """
    解析 PDF，提取文本和所有表格
    返回 JSON 结构
    """
    result = {
        "file_name": pdf_path.name,
        "text": [],
        "tables": {}
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

                # 提取表格
                tables = extract_tables_with_camelot(pdf_path, page_num)
                for idx, table in enumerate(tables, 1):
                    key = f"{pdf_path.stem}_table_{idx}_page_{page_num}"
                    result["tables"][key] = table

    except Exception as e:
        print(f"❌ 处理失败 {pdf_path.name}: {e}")
        return None

    return result


def main():
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        print(f"❌ PDF 目录不存在: {PDF_DIR}")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ {PDF_DIR} 下没有 PDF 文件")
        return

    for pdf_file in pdf_files:
        print(f"📄 正在处理: {pdf_file.name}")
        json_data = process_pdf(pdf_file)
        if json_data is None:
            continue

        out_file = Path(OUTPUT_DIR) / f"{pdf_file.stem}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存 JSON: {out_file.name}")


if __name__ == "__main__":
    main()
