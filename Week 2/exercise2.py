import os
import time
import asyncio
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

def run_invoke():
    print("\n===== SYNC invoke() — GROQ =====")
    groq_invoke_result = groq_chain.invoke({"topic": input})
    start_groq_invoke = time.time()
    print(f"\n[Time taken: {time.time() - start_groq_invoke:.2f}s]")
    print(groq_invoke_result)

    print("\n===== SYNC invoke() — GEMINI =====")
    gemini_result = gemini_chain.invoke({"topic": input})
    start_gemini_invoke = time.time()
    print(f"\n[Time taken: {time.time() - start_gemini_invoke:.2f}s]")
    print(gemini_result)

def run_stream():
    print("\n===== SYNC stream() — GROQ =====")
    groq_stream = time.time()
    for chunk in groq_chain.stream({"topic":input}):
        print(chunk,end='',flush=True)
    print(f"\n\n[Time taken: {time.time() - groq_stream:.2f}s]")

    print("\n===== SYNC stream() — GEMINI =====")
    gemini_stream = time.time()
    for chunk in gemini_chain.stream({"topic":input}):
        print(chunk,end='',flush=True)
    print(f"\n\n[Time taken: {time.time() - groq_stream:.2f}s]")

async def run_ainvoke():
    print("\n===== ASYNC ainvoke() — GROQ =====")
    start_ainvoke = time.time()
    result_ainvoke_groq = await groq_chain.ainvoke({"topic":input})
    print(result_ainvoke_groq)
    print(f"\n[Time taken: {time.time() - start_ainvoke:.2f}s]")

    print("\n===== ASYNC ainvoke() — GEMINI =====")
    start_ainvoke = time.time()
    result_ainvoke_gemini = await gemini_chain.ainvoke({"topic":input})
    print(result_ainvoke_gemini)
    print(f"\n[Time taken: {time.time() - start_ainvoke:.2f}s]")

async def run_astream():

    print("\n===== ASYNC astream() — Groq =====")
    start_groq_astream = time.time()
    async for chunk in groq_chain.astream({"topic":input}):
        print(chunk, end="", flush=True)
    print(f"\n\n[Time taken: {time.time() - start_groq_astream:.2f}s]")

    print("\n===== ASYNC astream() — GEMINI =====")
    start_gemini_astream = time.time()
    async for chunk in gemini_chain.astream({"topic":input}):
        print(chunk, end="", flush=True)
    print(f"\n\n[Time taken: {time.time() - start_gemini_astream:.2f}s]")


async def main():
    run_invoke()
    run_stream()
    await run_ainvoke()
    await run_astream()
    
asyncio.run(main())