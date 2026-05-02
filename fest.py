from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# ---------------- PAGE ----------------
st.set_page_config(page_title="QuickChat AI", page_icon="⚡")



search = GoogleSerperAPIWrapper()
    




# ---------------- PROMPT ----------------
system_prompt = """
You are an advanced AI assistant.

Rules:
- Answer general questions clearly and concisely
- If the question is about a document, use the provided context
- If the question requires current or real-time information, respond based on available knowledge
- If unsure, say so honestly
- Do not mention tools or internal logic
"""
# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile" , temperature=0.3
)

if "history" not in st.session_state:
    st.session_state.history = []

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "agent" not in st.session_state or st.session_state.agent is None:
    st.session_state.agent = create_agent(
        model=llm,
        tools=[search.run],  # always available
        checkpointer=st.session_state.memory,
        system_prompt=system_prompt
    )

# ---------------- PDF PROCESS ----------------
def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs = PyPDFLoader(path).load()
    os.unlink(path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(chunks, embeddings)
    return db


# ---------------- TOOLS ----------------
def create_rag_tool(retriever):

    @tool
    def rag_tool(query: str) -> str:
        """Search uploaded PDF documents."""
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant document context found."

        return "\n\n".join([d.page_content for d in docs])

    return rag_tool






# ---------------- RUN ----------------
def run_agent(query):

    agent = st.session_state.agent

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config={
            "configurable": {
                "thread_id": "quickchat-user"
            }
        }
    )

    return response["messages"][-1].content


# ---------------- UI ----------------
st.subheader("⚡ QuickChat AI")

# -------- SIDEBAR --------
with st.sidebar:
    st.header("📄 Upload PDF")

    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file:
        with st.spinner("Processing PDF..."):
            db = ingest_pdf(file)
            retriever = db.as_retriever(search_kwargs={"k": 3})

            # ✅ DYNAMIC TOOL SET
            tools = [search.run, create_rag_tool(retriever)]

            st.session_state.agent = create_agent(
                model=llm,
                tools=tools,
                checkpointer=st.session_state.memory,
                system_prompt=system_prompt
            )

        st.success("PDF Ready!")

# -------- CHAT HISTORY --------
for msg in st.session_state.history:
    st.chat_message(msg["role"]).markdown(msg["content"])

# -------- INPUT --------
query = st.chat_input("Ask Anything...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append(
        {"role": "user", "content": query}
    )

    with st.spinner("Thinking..."):
        answer = run_agent(query)

    st.chat_message("assistant").markdown(answer)

    st.session_state.history.append(
        {"role": "assistant", "content": answer}
    )