from pathlib import Path
import shutil

from fastapi import UploadFile

from app.schemas.document_schema import UploadResponse

from app.utils.document_reader import DocumentReader
from app.utils.text_cleaner import TextCleaner
from app.utils.recursive_chunker import RecursiveChunker

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService


class IngestionService:

    UPLOAD_FOLDER = "uploads"

    def __init__(self):

        self.chunker = RecursiveChunker()
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()

    def upload_document(self, file: UploadFile) -> UploadResponse:

        # Save uploaded file
        file_path = self._save_file(file)

        # Read document
        extracted_text = DocumentReader.read(file_path)

        # Clean extracted text
        clean_text = TextCleaner.clean(extracted_text)

        # Split into chunks
        chunks = self.chunker.split(clean_text)

        # Generate embeddings
        embeddings = self.embedding_service.generate_document_embeddings(chunks)

        # Store vectors in Pinecone
        self.vector_service.upsert_document(
            filename=file.filename,
            chunks=chunks,
            embeddings=embeddings
        )

        

        return UploadResponse(
            message="Document uploaded and stored successfully.",
            filename=file.filename,
            chunk_count=len(chunks)
        )

    def _save_file(self, file: UploadFile) -> str:

        Path(self.UPLOAD_FOLDER).mkdir(exist_ok=True)

        file_path = Path(self.UPLOAD_FOLDER) / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return str(file_path)