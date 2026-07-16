from fastapi import APIRouter,UploadFile,File

from app.services.ingestion_service import IngestionService
from app.schemas.document_schema import UploadResponse

document_router = APIRouter(prefix="/documents",tags=["Documents"])


@document_router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)):

    service = IngestionService()

    return service.upload_document(file)