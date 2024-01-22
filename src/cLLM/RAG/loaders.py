import os
from docx import Document
import fitz  # PyMuPDF


def read_pdf(file_path):
    with fitz.open(file_path) as pdf_document:
        text = ''
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            text += page.get_text()
    return text


def read_docx(file_path):
    doc = Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def read_files_in_directory(directory_path):
    result_dict = {}
    supported_extensions = {'.pdf', '.docx', '.txt', '.md'}

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        _, file_extension = os.path.splitext(file_name)

        if file_extension.lower() in supported_extensions:
            if file_extension.lower() == '.pdf':
                content = read_pdf(file_path)
            elif file_extension.lower() == '.docx':
                content = read_docx(file_path)
            elif file_extension.lower() in {'.txt', '.md'}:
                content = read_text_file(file_path)

            result_dict[file_name] = {'source': file_name, 'content': content}

    return result_dict
