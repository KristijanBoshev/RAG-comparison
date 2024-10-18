from app.naive.ingestion.document_ingest import Ingest
from app.settings import settings

file_path = settings.FILE_PATH

document_ingest = Ingest(
    file_path=file_path,
    extract_images=True,
)

__all__ = ["document_ingest"]