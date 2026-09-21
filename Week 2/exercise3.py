import os
import json
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

reviews = [
    "This phone's battery dies by 3 PM even with light use. Camera is great though.",
    "Absolutely love this laptop! Fast, lightweight, and the screen is gorgeous.",
    "The delivery took 3 weeks and the box arrived damaged. Product itself works fine.",
    "Customer support was rude and unhelpful when I asked about a refund.",
    "Decent headphones for the price, but the bass is weak and they feel cheap.",
    "Best purchase I've made this year. Setup was easy and it just works.",
    "App keeps crashing on Android, very frustrating. Wish it was more stable.",
    "Good value overall, though the instructions were confusing and in broken English.",
    "Terrible experience — item was completely different from what was advertised.",
    "Solid build quality, works as expected, no complaints so far. and also never recomanded to anyone",
]
groq = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv('GROQ_API_KEY').strip()
)

class ReviewAnalsis(BaseModel):
    """Structured analysis of a single customer review."""
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    key_issues: List[str] = Field(description="List of specific problems or complaints mentioned. Empty list if none.")
    summary: str = Field(description="A one-sentence summary of the review")
    rating_estimate: int = Field(description="estimated rating out of 5 for review", le=5, ge=1)

parser = PydanticOutputParser(pydantic_object=ReviewAnalsis)
template = ChatPromptTemplate.from_messages([
    ("system","You are an expert customer feedback analyst."
    "Analyze the given review and extract structured information.\n"
     "{format_instructions}"),
    ("human", "Review:\n{review}")
]).partial(format_instructions=parser.get_format_instructions())
groq_chain = template | groq | parser

input = [{"review": r} for r in reviews]
groq_result:List[ReviewAnalsis]=groq_chain.batch(input)

output_groq = []

for original_review, analysis in zip(reviews, groq_result):
    output_groq.append({
        "review": original_review,
        "sentiment":analysis.sentiment,
        "key_issues": analysis.key_issues,
        "summary": analysis.summary,
        "rating_estimate": analysis.rating_estimate,
    })

with open("review_analysis_results.json", "w") as f: json.dump( { "groq": output_groq}, f, indent=2 )

print(f"Saved {len(output_groq)} analyzed reviews to review_analysis_results.json")