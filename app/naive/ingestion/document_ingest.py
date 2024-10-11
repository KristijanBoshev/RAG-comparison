from langchain_community.document_loaders import PyPDFLoader
from app.settings import settings
from langchain_openai import OpenAIEmbeddings

"""
This class is responsible for ingesting a PDF file and splitting it into chunks of text.
"""
class Ingest:
    def __init__(self, file_path: str, extract_images: bool):
        self.file_path = file_path
        self.extract_images = extract_images
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
        )

    def _loader(self):
        loader = PyPDFLoader(
            file_path=self.file_path,
            extract_images=self.extract_images,
        )
        documents = loader.load_and_split()
        return documents
    
        
    def _embedding(self):
        split_docs = self._loader()

        embeddings = self.embedding_model.embed_documents(split_docs)
        return embeddings

