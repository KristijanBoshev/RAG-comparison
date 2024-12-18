from langchain_community.tools.tavily_search import TavilySearchResults
from app.settings import settings

user_agent = settings.USER_AGENT
web_search_tool = TavilySearchResults(k=3)