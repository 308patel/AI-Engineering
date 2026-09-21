import os
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

groq =  ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY').strip()
)

outputParser = CommaSeparatedListOutputParser()
format_instructions = outputParser.get_format_instructions()

template = ChatPromptTemplate([
    ("system","You have to provide only 5 tips for an topic. \n"
    "{format_instructions}"),
    ("human", "{topic}")
])

chain = template | groq | outputParser
results = chain.invoke({"topic":"AI","format_instructions": format_instructions})
print("======================== Groq Result ========================\n", results)
 