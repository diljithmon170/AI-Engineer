from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables from .env
load_dotenv()
app = FastAPI()

# Allow your frontend (running on any origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["http://localhost:5500"] if you want stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "FastAPI Q&A backend is running!"}

@app.post("/ask")
def ask_question(q: Question):
    # Demo answer (you can connect LLM or LangChain here later)
    api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key,
    )

    response = llm.invoke(q.question)
    return {"answer": f"{response.content}"}
