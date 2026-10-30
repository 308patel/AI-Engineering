import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

transactions = []

@tool
def add_income(source: str, amount: float, income: str) -> str:
    """Record an income entry with its source and amount."""
    transactions.append({
        "incomeType": income,
        "amt": amount, 
        "src": source
    })

    return f"transactions : {transactions}"

@tool
def add_expense(source: str, amount: float, income: str) -> str:
    """Record an expense entry with its category and amount."""
    transactions.append({
        "incomeType": income,
        "amt": amount, 
        "src": source
    })

    return f"expense is added in {transactions}"


@tool
def calculate_expense() -> str:
    """Get the total income, total expenses, and net balance."""
    totalExpense = 0
    totalIncome = 0
    print(">>>>>>>>>>>>>>>>>>>>>", transactions)
    for trn in transactions:
        if trn["incomeType"] == "expense":
            totalExpense += int(trn["amount"])
        elif trn["incomeType"] == "income":
            totalIncome += int(trn["amount"])
        else: 
            continue

    total = int(totalIncome) - int(totalExpense)
    return f'total net income is {total}'

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY").strip()
)

tools = [add_income, add_expense, calculate_expense]
agent = create_agent(
    model = llm,
    tools = tools
)

result = agent.invoke({
    "messages":[
        {
            "role":"user",
            "content":"i had 10000 income from salary, 20000 from stockmarket and 5000 from bonus also had 2000 as electricty bill, 5000 travelling cost, and 1000 for food calculate my remaining income"
            # "content":"40 deg celesius means how many fahrenheit? also 90 deg fahrenheit means how many celsius? and  67 kilometers means how many miles"
        }
    ]
})

# only purpose is to show use proper message for toolcalling
for msg in result['messages']:
    msg_type = type(msg).__name__
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"Tool Call: {tc['name']} and arguments is {tc['args']} ")
    if msg.content:
        print(f"Tool Call  >:: {msg.name}")
            

print("\n Final Answer : ", result["messages"][-1].content)