from model import groq_chain
from prompts import prompts

for prompt in prompts:
    print("======================= Groq Result =======================\n")
    response = groq_chain.invoke(prompt)
    print("\n",response)