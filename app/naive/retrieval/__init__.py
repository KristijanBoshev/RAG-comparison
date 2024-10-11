from app.naive.retrieval.vector_retrieval import Retrieval
from app.settings import settings

query = settings.QUERY

retrieval = Retrieval(query=query)

