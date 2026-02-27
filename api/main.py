from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ==============================
# 1️⃣ Set Gemini API Key
# ==============================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ==============================
# 2️⃣ Structured Output Schema
# ==============================
class StudyPlanOutput(BaseModel):
    plan: str = Field(description="Detailed step by step study plan")
    resources: List[str] = Field(description="List of recommended resources")
    tips: List[str] = Field(description="Practical study tips")


# ==============================
# 3️⃣ FastAPI
# ==============================
app = FastAPI(title="Tech Career Agent")


class UserInput(BaseModel):
    track: str
    level: str
    hours: int
    goal: str


@app.get("/")
def read_root():
    return {"message": "Look Ma, I'm deployed!"}


@app.post("/generate", response_model=StudyPlanOutput)
def generate(user_input: UserInput):

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional tech career mentor. "
         "Always respond with valid JSON only, no markdown, no extra text. "
         "The JSON must have exactly these keys: plan (string), resources (list of strings), tips (list of strings)."),
        ("human",
         "Generate a detailed study plan for:\n"
         "Track: {track}\n"
         "Level: {level}\n"
         "Hours per day: {hours}\n"
         "Goal: {goal}\n\n"
         "Respond ONLY with valid JSON.")
    ])

    chain = prompt | llm
    result = chain.invoke({
        "track": user_input.track,
        "level": user_input.level,
        "hours": user_input.hours,
        "goal": user_input.goal,
    })

    # Clean and parse the response
    text = result.content.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    parsed = json.loads(text)
    return StudyPlanOutput(**parsed)


# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)