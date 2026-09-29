import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["UNSTRUCTURED_YOLOX_MODEL_PATH"] = r"E:/models/yolox_l0.05.onnx"
# ====== 解析器 1: LlamaParse (云端) ======
try:
    from llama_parse import LlamaParse
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# ====== 解析器 2: Unstructured ======
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# ====== 解析器 3: MinerU ======
try:
    from mineru.tools.pdf_parser import PDFParser
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

# ====== 解析器 4: Marker ======
try:
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    MARKER_AVAILABLE = True
    # 预加载模型（加速后续调用）
    MARKER_MODELS = load_all_models()
except ImportError:
    MARKER_AVAILABLE = False


def save_result(output_dir: Path, base_name: str, parser_name: str, data: dict):
    """统一保存格式：{base_name}_{parser}.json"""
    out_path = output_dir / f"{base_name}_{parser_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 保存 {parser_name} 结果: {out_path.name}")


def parse_with_llama(pdf_path: Path, output_dir: Path):
    if not LLAMA_AVAILABLE or not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("  ❌ LlamaParse 未配置或不可用")
        return

    parser = LlamaParse(
        result_type="markdown",
        language="ch_sim",
        user_prompt="这是中国上市公司财报，请准确提取所有文字段落和表格，保留原始结构。",
        invalidate_cache=True,
        verbose=False
    )
    try:
        docs = parser.load_data(str(pdf_path))
        if not docs:
            raise ValueError("Empty response")
        md_text = docs[0].text

        # 粗略拆分（你可在此处增强表格结构化）
        data = {
            "source": str(pdf_path),
            "full_markdown": md_text,
            "paragraphs": [p.strip() for p in md_text.split("\n\n") if p.strip()],
            "tables": []  # 可后续用 pandas 从 markdown 提取
        }
        save_result(output_dir, pdf_path.stem, "llama", data)
    except Exception as e:
        print(f"  ❌ LlamaParse 失败: {e}")


def parse_with_unstructured(pdf_path: Path, output_dir: Path):
    if not UNSTRUCTURED_AVAILABLE:
        print("  ❌ Unstructured 未安装")
        return

    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            # model_path=r"E:/model/yolox_l0.05.onnx",
            languages=["chi_sim"],  # 需 tesseract 支持中文
            extract_images_in_pdf=False,
        )
        paragraphs = []
        tables = []
        for el in elements:
            if el.category == "Table":
                tables.append({"text": el.text})  # el.text 是 tab 分隔
            else:
                paragraphs.append(el.text)

        data = {
            "source": str(pdf_path),
            "paragraphs": paragraphs,
            "tables": tables
        }
        save_result(output_dir, pdf_path.stem, "unstructured", data)
    except Exception as e:
        print(f"  ❌ Unstructured 失败: {e}")


def parse_with_mineru(pdf_path: Path, output_dir: Path):
    if not MINERU_AVAILABLE:
        print("  ❌ MinerU 未安装")
        return

    try:
        parser = PDFParser()
        # MinerU 输出为 Markdown + 表格 JSON
        result = parser(pdf_path, output_dir=str(output_dir), save_mode="markdown")
        # MinerU 会生成 .md 和 .json，我们读取 .json
        json_path = output_dir / f"{pdf_path.stem}.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # MinerU 的 JSON 已结构化表格
            save_result(output_dir, pdf_path.stem, "mineru", data)
        else:
            # 退化为读取 markdown
            md_path = output_dir / f"{pdf_path.stem}.md"
            if md_path.exists():
                with open(md_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
                data = {"source": str(pdf_path), "full_markdown": md_text}
                save_result(output_dir, pdf_path.stem, "mineru", data)
    except Exception as e:
        print(f"  ❌ MinerU 失败: {e}")


def parse_with_marker(pdf_path: Path, output_dir: Path):
    if not MARKER_AVAILABLE:
        print("  ❌ Marker 未安装")
        return

    try:
        md_text, _, _ = convert_single_pdf(
            str(pdf_path),
            MARKER_MODELS,
            langs=["zh"]  # Marker 支持 "zh"
        )
        data = {
            "source": str(pdf_path),
            "full_markdown": md_text,
            "paragraphs": [p.strip() for p in md_text.split("\n\n") if p.strip()]
        }
        save_result(output_dir, pdf_path.stem, "marker", data)
    except Exception as e:
        print(f"  ❌ Marker 失败: {e}")


def main():
    input_dir = Path("E:/model/report")
    output_dir = input_dir / "parsed_results"
    output_dir.mkdir(exist_ok=True)

    for pdf_path in input_dir.glob("*.pdf"):
        print(f"\n🔍 正在对比解析: {pdf_path.name}")
        
        # 为每个解析器创建独立输出（避免冲突）
        # parse_with_llama(pdf_path, output_dir)
        parse_with_unstructured(pdf_path, output_dir)
        # parse_with_mineru(pdf_path, output_dir)
        # parse_with_marker(pdf_path, output_dir)


if __name__ == "__main__":
    main()