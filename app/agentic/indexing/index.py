from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class Index:
    def __init__(self, urls, collection_name, chunk_size=500, chunk_overlap=150):
        self.urls = urls
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore = None
        self.retriever = None
    
    def load_documents(self):
        docs = [WebBaseLoader(url).load() for url in self.urls]
        docs_list = [item for sublist in docs for item in sublist]
        return docs_list
    
    def split_documents(self, docs_list):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_splits = text_splitter.split_documents(docs_list)
        return doc_splits
    
    def create_vectorstore(self, doc_splits):
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=self.collection_name,
            embedding=OpenAIEmbeddings(),
        )
        return self.vectorstore
    
    def create_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore has not been initialized. Call `create_vectorstore()` first.")
        self.retriever = self.vectorstore.as_retriever()
    
        return self.retriever
    
    def workflow(self):
        docs_list = self.load_documents()
        doc_splits = self.split_documents(docs_list)
        self.create_vectorstore(doc_splits)
        retriever = self.create_retriever()

        return retriever