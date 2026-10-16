import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

groq = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY").strip()
)

template = ChatPromptTemplate([
    ("system", "You are an expert in movie reviews"),
    ("human", "Review this movie: {review}")
])

output_parser = StrOutputParser()

groq_chain = template | groq | output_parser
