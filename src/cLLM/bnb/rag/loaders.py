import os
from docx import Document
import fitz  # PyMuPDF

import re


def read_pdf(file_path):
    with fitz.open(file_path) as pdf_document:
        text = ""
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()
    return text


def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def read_files_in_directory(directory_path):
    result_dict = {}
    supported_extensions = {".pdf", ".docx", ".txt", ".md"}

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() in supported_extensions:
            if file_extension.lower() == ".pdf":
                content = read_pdf(file_path)
            elif file_extension.lower() == ".docx":
                content = read_docx(file_path)
            elif file_extension.lower() in {".txt", ".md"}:
                content = read_text_file(file_path)
            else:
                raise RuntimeError(f"Unknown file extension of {file_extension.lower()}")
            result_dict[file_name] = {"source": file_name, "content": content}

    return result_dict


def split_into_chunks(text, chunk_size, overlap_size):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    chunks = []
    current_chunk = ""
    current_chunk_size = 0

    for sentence in sentences:
        sentence_size = len(sentence)

        if current_chunk_size + sentence_size <= chunk_size:
            current_chunk += sentence + " "
            current_chunk_size += sentence_size
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_chunk_size = sentence_size - overlap_size

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
