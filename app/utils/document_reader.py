import fitz
from docx import Document
from pathlib import Path


class DocumentReader:
    """
    Reads supported document formats and returns plain text.
    """

    @staticmethod
    def read(file_path: str) -> str:

        extension = Path(file_path).suffix.lower()

        if extension == ".pdf":
            return DocumentReader._read_pdf(file_path)

        elif extension == ".docx":
            return DocumentReader._read_docx(file_path)

        elif extension == ".txt":
            return DocumentReader._read_txt(file_path)

        else:
            raise ValueError(f"Unsupported file type: {extension}")

    @staticmethod
    def _read_pdf(file_path: str) -> str:

        document = fitz.open(file_path)

        text = ""

        for page in document:
            text += page.get_text()

        document.close()

        return text

    @staticmethod
    def _read_docx(file_path: str) -> str:

        document = Document(file_path)

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    @staticmethod
    def _read_txt(file_path: str) -> str:

        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()