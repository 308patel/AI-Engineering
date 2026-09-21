from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import os     
from dotenv import load_dotenv
load_dotenv("../.env")
prompt = "Explain what a REST API is in 3 sentences."
print(">>>>>>>>>>>>>>>>>>>>>",os.getenv("GROQ_API_KEY").strip())
print(">>>>>>>>>>>>>>>>>>>>>",os.getenv("GOOGLE_API_KEY").strip())
groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY").strip(),
)

gemini = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    max_tokens=512,
    api_key=os.getenv("GOOGLE_API_KEY").strip()
)

response = groq.invoke(prompt)
geminiResp = gemini.invoke(prompt)
print("------------------------GROQ------------------------ \n",response)
print("------------------------GEMINI------------------------ \n ",geminiResp)