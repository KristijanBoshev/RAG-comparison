from app.naive.generation.generate_answer import Generation
from langchain_openai import ChatOpenAI
from app.settings import settings

llm = ChatOpenAI(model=settings.MODEL, api_key=settings.OPENAI_API_KEY)
query = settings.QUERY

generation = Generation(query=query, llm=llm)

__all__ = [
    "generation", "Generation"
]