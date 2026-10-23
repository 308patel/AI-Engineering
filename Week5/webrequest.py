from langchain_community.tools import RequestsGetTool
from langchain_community.utilities import TextRequestsWrapper

request_tool = RequestsGetTool(requests_wrapper=TextRequestsWrapper(),allow_dangerous_requests=True)

result = request_tool.run("https://www.google.com/")
print("result ::", result)