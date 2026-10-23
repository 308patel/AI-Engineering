from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
arxivWrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)

arxiv_tool = ArxivQueryRun(api_wrapper=arxivWrapper)

result = arxiv_tool.run("AI Engineering in 2026")
print("result ::\n", result)