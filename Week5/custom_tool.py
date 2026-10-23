from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
import os
from dotenv import load_dotenv
load_dotenv()
# creating tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

#setup llm
llm = ChatGroq(
    model_name="openai/gpt-oss-120b", 
    temperature=0, 
    api_key=os.getenv("GROQ_API_KEY").strip()
)

tools = [add, multiply]

# bind tools with llm
llm_with_tools = llm.bind_tools(tools)

# create a tool lookup dictionary
tool_lookup = {t.name: t for t in tools}

message = [HumanMessage("add 4 and 7 also multiply 56 and 76")]
while True: 
    response = llm_with_tools.invoke(message)

    print("Response:", response.tool_calls)

    message.append(response)
    if not response.tool_calls:
        print("✅ Final Answer:", response.content)
        break  # Exit the loop


    for tool_call in response.tool_calls:
        selected_tool = tool_lookup[tool_call["name"]]

        tool_result = selected_tool.invoke(tool_call["args"])

        message.append(
            ToolMessage(
                content = str(tool_result),
                tool_call_id = tool_call["id"],
            )
        )

    final_response = llm_with_tools.invoke(message)
    print("Final Response:", final_response.content)

