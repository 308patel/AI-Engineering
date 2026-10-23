
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()

no = input("Please \n Enter 1 to use Duck Suck Go Search \n Enter 2 to use AI Agent Search(Tavily)")

if(no == "1"):
    search = DuckDuckGoSearchRun()

    result = search.run("Who is Ryan Dhal?")
    print(result)

elif(no == "2"):

    search = TavilySearchResults(max=3, api_key=os.getenv("TAVILY_API_KEY"))
    result = search.run("Tips to learn AI Engineering in 2026")
    print(result)
else:
    print("Invalid input. Please enter 1 or 2.")