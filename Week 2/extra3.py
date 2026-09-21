from distro import name
import os
from langchain_groq import ChatGroq
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

groq = ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=os.getenv("GROQ_API_KEY").strip()
)

class Recipe(BaseModel):
    name:str= Field(description="Name of dish")
    ingredients:List[str] = Field(description="list of ingredients used in dish")
    prep_time_minutes: int = Field(description="time to compelet making of dish", gt=0)


parser = PydanticOutputParser(pydantic_object=Recipe)
template = ChatPromptTemplate.from_messages([
    ("system","You are an headchef in 7 start hotel.analyze the recipe and extract structured output  \n"
    "{format_instructions}"),
     ("human", "Recipe:\n{recipe}")
]).partial(format_instructions=parser.get_format_instructions())

groq_chain = template | groq | parser

input = "A refreshing Mango Smoothie is perfect for hot days. Just blend ripe mango, plain yogurt, honey, a splash of milk, and a few ice cubes. It only takes about 5 minutes to prepare."
result = groq_chain.invoke({"recipe":input})
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n",result)
