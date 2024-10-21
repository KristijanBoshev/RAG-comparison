from app.graph.generate.generate import Generation
from langchain_openai import ChatOpenAI
from app.settings import settings

model = settings.MODEL
api_key = settings.OPENAI_API_KEY

llm = ChatOpenAI(model = model, api_key=api_key)
generate = Generation(llm = llm)

__all__ = ["generate"]