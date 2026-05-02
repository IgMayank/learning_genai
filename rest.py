from dotenv import load_dotenv
load_dotenv()

import os, tempfile
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="llama-3.3-70b-versatile")
search_wrapper = GoogleSerperAPIWrapper()

SYSTEM = SystemMessage(content="""You are a powerful AI assistant with access to web search and uploaded documents.
- For greetings and casual conversation, respond naturally and friendly like a human.
- If the user shares personal info like their name, acknowledge it warmly and remember it in the conversation.
- For questions about uploaded documents, answer strictly from the document context provided.
- For factual or current questions, use the web search results provided.
- Never search the web or documents for casual conversation or personal statements.
- Always respond in a clear, confident, and helpful tone.""")

CASUAL = ["hi", "hii", "hiii", "hello", "hey", "how are you", "how r u", "what's up", "sup", "good morning", "good evening", "good night"]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []
    st.session_state.retriever = None

def ingest_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    docs = PyPDFLoader(tmp_path).load()
    os.unlink(tmp_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

def search_docs(query):
    docs = st.session_state.retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    return llm.invoke([SYSTEM, HumanMessage(content=f"Answer using this context:\n{context}\n\nQuestion: {query}")]).content

def run_agent(query):
    history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history[-10:]])
    
    prompt = f"""Previous conversation:
{history_context}

User: {query}"""

    # First ask LLM if it can answer on its own
    check = llm.invoke([
        SystemMessage(content="You are a decision maker. Reply with only YES if you can confidently answer the question from your own knowledge or the conversation history. Reply with only NO if you need to search the web for current or specific information."),
        HumanMessage(content=query)
    ]).content.strip().upper()

    if check == "NO":
        if st.session_state.retriever:
            return search_docs(query)
        return search_wrapper.run(query)
    
    return llm.invoke([SYSTEM, HumanMessage(content=prompt)]).content

st.subheader("⚡Quickchat - Answers as you question 😊")

with st.sidebar:
    st.header("📄 Upload Document (RAG)")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            vs = ingest_pdf(uploaded_file)
            st.session_state.retriever = vs.as_retriever(search_kwargs={"k": 3})
        st.success("PDF ready!")

for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

query = st.chat_input("Ask Anything?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    answer = run_agent(query)
    st.chat_message("ai").markdown(answer)
    st.session_state.history.append({"role": "ai", "content": answer})
