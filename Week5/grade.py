from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
load_dotenv()

class StudentGrade(BaseModel):
    name: str
    average: float

@tool
def get_student_marks(student_name:str) ->str:
    """get the marks of student by name"""
    students = {
        "alice": {"math": 85, "science": 90, "english": 78},
        "bob":   {"math": 60, "science": 55, "english": 70},
        "carol": {"math": 95, "science": 88, "english": 92},
    }
    return students.get(student_name.lower(), f"student with name {student_name} is not found")

@tool
def calculate_average(marks: str) -> str:
    """calculate total average of student from comma seprated values. Example input: '85,90,78'"""
    marks_list = marks.split(",")
    values = [int(mark.strip()) for mark in marks_list]
    return sum(values) / len(values)

@tool
def calculate_grade(average: float) ->str:
    """calculate grade based on average"""
    if(average >= 90):
        return "A"
    elif(average>=80):
        return "B"
    elif(average>=70):
        return "C"
    elif(average>=60):
        return "D"
    else:
        return "F"

llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY").strip() 
)

tools = [get_student_marks, calculate_average, calculate_grade]

agent=create_agent(
    model=llm,
    tools=tools,
    response_format=StudentGrade
)

result = agent.invoke({
    "messages":[
        {
            "role": "user",
            "content": "alice, bob or carol whos average is below average 70"
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
            
# print("\n✅ Final Answer:",result["messages"][-1].content)
print(">>>>>.", result["structured_response"] )