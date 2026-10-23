from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
load_dotenv()


@tool
def get_product_price(product: str) -> str:
    """Get the current price of a product from a store"""
    prices = { 
        "laptop": "99999",
        "phone": "49999",
        "headphones": "1999",
    }
    return prices.get(product.lower(), "Product not found")

@tool
def get_product_discount(price: int, discount: int) -> str:
    """Calculate price after applying discount"""

    discounted_price = price - (price * discount / 100)
    return f"The price after {discount}% discount is: {discounted_price}"

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY").strip()
)

tools = [get_product_price, get_product_discount, add, multiply]


agent = create_agent(
    model = llm,
    tools = tools,
)

# agent_executor = AgentExecutor(
#     agent=agent, 
#     tools=tools, 
#     verbose=True, 
#     handle_parsing_errors=True
# )

result = agent.invoke({
    "messages":[
        {
            "role": "user",
            "content": "What is price of laptop and add 1000 per laptop(one laptop -> 1000, 2 laptops -> 2000) as extended warranty charge and quantity of laptop is 4 and overall discount is 49%"
        }
    ]
})

for msg in result['messages']:
    msg_type = type(msg).__name__
    if hasattr(msg, 'tools_calls') and msg.tools_calls:
        for tc in msg.tools_calls:
            print(f"Tool Call: {tc['name']} with args: {tc['args']}")
    if msg.content:
        # print(f"{msg_type}: {msg.content}")
        continue
print("\n✅ Final Answer:",result["messages"][-1].content)