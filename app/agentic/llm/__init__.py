from app.agentic.llm.generate import Generate
from app.settings import settings

generate = Generate(model = settings.MODEL, temperature=0)

__all__ = ["generate"]