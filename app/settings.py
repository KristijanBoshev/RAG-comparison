from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()


class Settings(BaseSettings):
    # Key configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # Query configuration
    QUERY: str = "Write comprehensive answer on what is multi-head attention and main benefits using it"
   
    # File configuration
    FILE_PATH: str = "/Users/kristijanboshev/Library/GitHub/RAG-comparison/data/Attention.pdf"

    # Chroma configuration
    CHROMA_COLLECTION_NAME: str = 'collection'
    CHROMA_PERSIST_DIRECTORY: str = './chroma_db'

    # LLM configuration
    MODEL: str = 'gpt-4o-mini'
    
    # Search engine
    SEARCH_ENGINE: str = 'TAVILY'
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY")
    URLS: List[str] = ["https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853",
            "https://medium.com/@sachinsoni600517/multi-head-attention-in-transformers-1dd087e05d41",
            ]
    
    # Neo4j credentials
    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USERNAME:str = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")
    AURA_INSTANCEID: str = os.getenv("AURA_INSTANCEID")
    AURA_INSTANCENAME:str = os.getenv("AURA_INSTANCENAME") 

    class Config:
        env_file_encoding = "utf-8"  # Ensure the env file is read with UTF-8 encoding
        
        


settings = Settings()


