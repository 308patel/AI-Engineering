from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

model = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY").strip() 
)
async def main():

    # Creating MCP Client
    client = MultiServerMCPClient({
        # Weather MCP Server
        "weather": {
            "transport": "streamable_http",
            "url": "https://weather.chukai.io/mcp"   # ← public HTTP server
        }
    })

    # System Prompt
    system_prompt=system_prompt = """
You are a tool-dependent assistant. You are strictly prohibited from answering
using your own pretrained knowledge.

Rules:
1. Do not answer from pretrained knowledge, memory, assumptions, or independent calculations.
2. If no available tool can answer the question, respond exactly: 'i dont know'
3. If the tool output does not contain sufficient information to answer,
4. respond exactly: 'i dont know'
5. Never fabricate, assume, or supplement information missing from tool outputs.
6. Your final answer must be directly supported by the tool output.
7. Do not repeatedly call tools for the same operation.
8. Answer in 4-5 words for weather data.
"""

    # Get available tools
    tools = await client.get_tools()
    print("Available tool list is \n")
    for tool in tools:
        print("Tool Name :",tool.name)
    selected_tools = [
    tool for tool in tools
    if tool.name in [
        "get_weather_forecast",
        "get_air_quality"
    ]
]
    # Create agent
    agent = create_agent(model, tools=selected_tools, system_prompt=system_prompt)

    # Invoke agent
    result = await agent.ainvoke({
        "messages":[{
            "role":"user", "content": "What is weather in Delhi?"
        }]
    })
    print("\n \n",result['messages'][-1].content)
asyncio.run(main())