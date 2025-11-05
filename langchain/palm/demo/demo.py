# --- Create a memory object ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# --- Initialize the LLM ---
llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=api_key,
    )

# --- 2️⃣ Create memory (store messages manually) ---
memory = []

# --- 3️⃣ Define a dynamic chat prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that remembers the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# --- 4️⃣ Build runnable chain ---
chain = RunnableSequence(prompt | llm)

# --- 5️⃣ Start chat loop ---
print("🤖 Groq Chatbot Ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Run the chain with full history
    response = chain.invoke({
        "history": memory,
        "input": user_input
    })

    print("Bot:", response.content)

    # --- Save messages into memory (for context) ---
    memory.append(HumanMessage(content=user_input))
    memory.append(AIMessage(content=response.content))
print(memory)
