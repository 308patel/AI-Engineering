from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
import os
load_dotenv()

cart= []

@tool
def search_product(product_name: str) ->str:
    """search for product and return price"""
    products = {
        "apple":    10,
        "bread":    35,
        "milk":     55,
        "eggs":     80,
        "cheese":  120,
    }
    return products.get(product_name.lower(), "Product not found")

@tool
def add_to_cart(products:str, qty:int, price:int)-> str:
    """add a product with given qty to the cart"""
    return cart.append({"name":products,"qty": qty, "price":price})

@tool
def calculate_cart_total() -> str:
    """Calculate and return the total price of all items in the cart."""
    total= (sum(int(pro.qty) * int(pro.price)) for pro in cart)
    return total


llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY").strip() 
)

tools = [search_product, add_to_cart, calculate_cart_total]


agent=create_agent(
    model=llm,
    tools=tools
)
result = agent.invoke({
    "messages":[
        {
            "role": "user",
            "content": "add 10 apple, 20 eggs  in cart and provide total"
        }
    ]
})

for msg in result['messages']:
    msg_type = type(msg).__name__
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"Tool Call: {tc['name']} and arguments is {tc['args']} ")
    if msg.content:
        print(f"Tool Call  >:: {msg.name}")
            
print("\n✅ Final Answer:",result["messages"][-1].content)