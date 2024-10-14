from app.agentic.indexing.index import Index
from app.settings import settings


index = Index(urls=settings.URLS, collection_name="agentic")

__all__ = ["index"]