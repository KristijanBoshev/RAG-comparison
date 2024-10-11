from app.naive.ingestion.document_ingest import Ingest


document_ingest = Ingest(
    file_path="data/Attention.pdf",
    extract_images=True,
)

__all__ = ["document_ingest"]