import os

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 
load_dotenv()
groq = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY').strip()
)

gemini = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=os.getenv('GOOGLE_API_KEY').strip()
)

template = ChatPromptTemplate([
    ("system","You are node js expoert with 20+ year of experiance in backend development"),
    ("human","Explain {topic}")
])

parser=StrOutputParser()
input = "Explain node js in one line"

groq_chain = template | groq | parser
gemini_chain = template | gemini | parser

print("------------------------------------------------GROQ------------------------------------------------\n", groq_chain.invoke({"topic":input}))
print("------------------------------------------------GEMINI------------------------------------------------\n", gemini_chain.invoke({"topic":input}))