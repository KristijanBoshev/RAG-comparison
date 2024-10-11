from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from a .env file (optional, if not already loaded elsewhere)
load_dotenv()


class Settings(BaseSettings):
    # Key configuration
    OPENAI_API_KEY: str

    # Query configuration
    QUERY: str = "Write comprehensive answer on what is multi-head attention and main benefits using it"
   
    # File configuration
    FILE_PATH: str = "data/Attention.pdf"

    # Chroma configuration
    CHROMA_COLLECTION_NAME: str = 'collection'
    CHROMA_PERSIST_DIRECTORY: str = './chroma_db'

    # LLM configuration
    MODEL: str = 'gpt-4o-mini'

    class Config:
        env_file_encoding = "utf-8"  # Ensure the env file is read with UTF-8 encoding


# Initialize the settings object
settings = Settings()

# Accessing the settings:
# You can now access the values using settings.OPENAI_API_KEY, settings.MODEL, etc.
