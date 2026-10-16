from langchain_core.runnables import RunnableParallel
from langchain_core.tools import retriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv 

load_dotenv()

summary_template = ChatPromptTemplate([
    ("system", "you are expert in summarize paragraph"),
    ("human", "{paragraph}")
])

groq = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY").strip()
)

outputParser = StrOutputParser()

sentiment_template = ChatPromptTemplate([
    ("system", "you are expert in reading sentiments"),
    ("human", "{sentiment}")
])

summary_chain = summary_template | groq | outputParser 
sentiment_chain = sentiment_template | groq | outputParser

final_chain = RunnableParallel(
    original = RunnablePassthrough(),
    summary = summary_chain,
    sentiment = {"sentiment" : RunnablePassthrough() } | sentiment_chain
)
result = final_chain.invoke({"paragraph":"The new fitness tracker I bought last month has completely changed my morning routine. The battery lasts nearly a week, which is way better than my old device, and the app is surprisingly easy to use. My only complaint is that the strap started fraying after just a few weeks of regular use, which feels disappointing for a premium-priced product."})

print(":::::::::::::::::::::::::::::::::::::::::: summary :::::::::::::::::::::::::::::::::::::::::: \n", result["summary"])
print(":::::::::::::::::::::::::::::::::::::::::: sentiment :::::::::::::::::::::::::::::::::::::::::: \n", result["sentiment"])
print(":::::::::::::::::::::::::::::::::::::::::: original :::::::::::::::::::::::::::::::::::::::::: \n",result["original"])
# print(".......................",result["original"]["summary"])

