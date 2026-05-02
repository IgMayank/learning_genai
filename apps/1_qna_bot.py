import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

st.title("Ask Buddy-Your ChatBot ")
st.markdown("A very basic chat Bot with the help of google gemini and langchain")
query = st.chat_input("Ask anything")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role =  message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

if query:
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({"role":"ai","content":res.content}) 