from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import wikipedia
wikipedia.set_user_agent(
    "ai-engineering-learning/1.0 (your-email@example.com)"
)
wikipediaWrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=500
)

wiki_tool = WikipediaQueryRun(api_wrapper=wikipediaWrapper)

result = wiki_tool.invoke("RG Sharma")
print("result ::\n", result)