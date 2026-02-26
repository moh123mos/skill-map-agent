from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.output_parsers import JsonOutputParser


# ==============================
# 1️⃣ Set Gemini API Key
# ==============================
# حطي مفتاحك هنا أو في environment variable
os.environ["GOOGLE_API_KEY"] = "AIzaSyBiw5o4Zv5t40PS8oB7naNeDqt3UD157nQ"


# ==============================
# 2️⃣ Define LLM (Gemini)
# ==============================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)


# ==============================
# 3️⃣ Structured Output Schema
# ==============================
class StudyPlanOutput(BaseModel):
    plan: str = Field(description="Detailed step by step study plan")
    resources: List[str] = Field(description="List of recommended resources")
    tips: List[str] = Field(description="Practical study tips")


parser = JsonOutputParser(pydantic_object=StudyPlanOutput)


# ==============================
# 4️⃣ Tool
# ==============================
@tool
def generate_study_plan(track: str, level: str, hours: int, goal: str):
    """
    Generate structured study plan for a tech track.
    """

    prompt = f"""
    Create a structured learning plan.

    Track: {track}
    Level: {level}
    Study hours per day: {hours}
    Goal: {goal}

    Return response in JSON format with:
    plan: string
    resources: list of strings
    tips: list of strings
    """

    response = llm.invoke(prompt)
    return response.content


# ==============================
# 5️⃣ Create Agent
# ==============================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional tech career mentor."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, [generate_study_plan], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[generate_study_plan], verbose=True)


# ==============================
# 6️⃣ FastAPI
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

    user_prompt = f"""
    Generate study plan for:
    Track: {user_input.track}
    Level: {user_input.level}
    Hours per day: {user_input.hours}
    Goal: {user_input.goal}
    """

    result = agent_executor.invoke({"input": user_prompt})

    # Parse structured JSON
    parsed = parser.parse(result["output"])

    return parsed

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)