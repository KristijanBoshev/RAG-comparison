from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

class Convertor:
    def __init__(self, file_path, extract_images, llm):
        self.file_path = file_path
        self.extract_images = extract_images
        self.llm = llm
    
    def _loader(self):
        loader = PyPDFLoader(
            file_path=self.file_path,
            extract_images=self.extract_images,
        )
        documents = loader.load_and_split()
        return documents
    
    def _convert(self):
        documents = self._loader()
        llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            verbose=True,
        )
        graph_documents=llm_transformer.convert_to_graph_documents(documents)
        return graph_documents
    
    
        
    
    

