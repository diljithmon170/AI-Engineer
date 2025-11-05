
import streamlit as st
from langchain_code import LangchainCode


def create_chatbot_interface(llm):
    st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")

    # --- Title ---
    st.title("💬 Chatbot Interface")

    # --- Initialize chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display chat messages ---
    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # --- Chat input box ---
    question = st.chat_input("Type your message...")

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate a demo response (you can replace with API call)
        response = llm.get_groq_response(question)
        bot_response = f"'{response}'"

        # Add bot message
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    langchain_code = LangchainCode()
    create_chatbot_interface(langchain_code)