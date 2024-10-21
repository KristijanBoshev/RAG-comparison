from app.graph.convert.convert import Convertor
from app.settings import settings
from langchain_openai import ChatOpenAI

path = settings.FILE_PATH
api_key = settings.OPENAI_API_KEY
model = settings.MODEL

llm = ChatOpenAI(api_key=api_key, model=model)
convert = Convertor(file_path=path, extract_images=True, llm=llm)

__all__ = ["convert"]