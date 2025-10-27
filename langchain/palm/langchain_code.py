from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class LangchainCode:
    def __init__(self):
        self.llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key,
    )

    def get_groq_response(self, question: str) -> str:
        response = self.llm.invoke(question)
        return response.content
