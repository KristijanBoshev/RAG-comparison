from app.graph.KG.knowledge_graph import KGraph
from app.settings import settings

uri = settings.NEO4J_URI
username = settings.NEO4J_USERNAME
password = settings.NEO4J_PASSWORD


kgraph = KGraph(uri=uri, username=username, password=password)

__all__ = ["kgraph"]