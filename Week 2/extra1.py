from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv 

load_dotenv()

template = ChatPromptTemplate([
    ("system", "you are vocalbury expert give meanging of every words"),
    ("human", "Give only 1 popular synonym for {word}")
])

groq = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY").strip()
)

outputParser = StrOutputParser()

chain = template | groq | outputParser

print("======================== Groq Result ========================\n", chain.invoke({"word":"price"}))