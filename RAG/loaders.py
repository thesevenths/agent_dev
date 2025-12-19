from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader
import os

def load_pdf_with_tables(pdf_dir: str) -> list[Document]:
    """
    使用 PyMuPDFReader 加载 PDF，并保留表格结构（以文本形式）
    LlamaIndex 默认的 PDFReader 会尝试保留表格格式（如转为 Markdown）
    """
    pdf_files = [str(p) for p in Path(pdf_dir).glob("*.pdf")]
    reader = PyMuPDFReader()
    documents = []
    for pdf_file in pdf_files:
        try:
            docs = reader.load_data(file=pdf_file)
            # 添加元数据
            for doc in docs:
                doc.metadata["file_path"] = pdf_file
                doc.metadata["file_name"] = os.path.basename(pdf_file)
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {pdf_file}: {e}")
    return documents