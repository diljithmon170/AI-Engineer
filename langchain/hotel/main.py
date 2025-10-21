import streamlit as st
import llm
st.title("Hotel Name and Dishes Extraction")

st.write("This application extracts hotel names and dishes from user reviews using LangChain.")

cusine = st.sidebar.selectbox("select cusine", ["Indian", "Chinese", "Italian", "Mexican"])

if cusine:

    st.write(f"You selected {cusine} cuisine.")
    response = llm.generate_hotel_info(cusine)
    st.header(response['hotel_name'])
    menu_items = response['menu_items'].split(",")
    st.write("### Dishes mentioned in the review:")
    for item in menu_items:
        st.write(f"- {item.strip()}")