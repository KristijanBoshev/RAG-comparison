from langchain_chroma import Chroma
from app.settings import settings
from app.naive.ingestion import document_ingest

collection_name = settings.CHROMA_COLLECTION_NAME
persistent_directory = settings.CHROMA_PERSIST_DIRECTORY

class Retrieval:
    def __init__(self, query):
        self.query = query

    def _create_vector_store(self):
        vector_store = Chroma.from_documents(
            collection_name=collection_name,
            persist_directory=persistent_directory,
            documents= document_ingest._loader(),
            embedding=document_ingest.embedding_model
            
        )
        return vector_store
    
    def _assign_retriever(self):
        vector_store = self._create_vector_store()
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 30}
        )
        docs = retriever.invoke(self.query)
        return docs

