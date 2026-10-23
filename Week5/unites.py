import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

@tool
def km_to_miles(km: float) -> float:
    """Convert kilometers to miles."""
    return km * 0.621371

@tool
def kg_to_pounds(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462

@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """convert celsius to farehenit"""
    return (celsius * 9/5) + 32

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY").strip()
)

tools = [km_to_miles, kg_to_pounds, celsius_to_fahrenheit]

agent = create_agent(
    model = llm,
    tools = tools
)

result = agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":"Convert 10 km to miles, 70 kg to pounds, and 100°C to Fahrenheit"
            # "content":"40 deg celesius means how many fahrenheit? also 90 deg fahrenheit means how many celsius? and  67 kilometers means how many miles"
        }
    ]
})

# only purpose is to show use proper message for toolcalling
for msg in result["messages"]:
    msg_type = type(msg).__name__
    if hasattr(msg, 'tools_calls') and msg.tools_calls:
        for tc in msg.tools_calls:
            print(f"Tool Call: {tc['name']} with args: {tc['args']}")
    if msg.content:
        continue

    print("\n Final Answer : ", result["messages"][-1].content)