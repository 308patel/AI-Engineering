import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client,StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()


async def main():
    server_param = StdioServerParameters(
        command="python",
        args=["math_server.py"]
    )

    model = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY").strip() 
    )

    async with stdio_client (server_param) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")

            agent = create_agent(
                model,
                tools
            )

            result = await agent.ainvoke({
                "messages": [HumanMessage(content="What is 35 and 35?")]  
            })

            print("Result : ",result['messages'][-1].content)
asyncio.run(main())