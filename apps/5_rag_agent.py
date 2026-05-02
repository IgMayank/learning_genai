from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader , PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
llm_new = ChatGroq(model="openai/gpt-oss-20b",streaming=True)


memory = InMemorySaver()


# st session 
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = [ ]


def process_document(path):

    # Load the document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    # Split into chunks
    splitter  = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap=600)
    docs = splitter.split_documents(documents=docs)

    # embeddings and vector db 

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )
    # create a agent - tool , llm ,systemprompt

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",convert_system_message_to_human=True)



    @tool
    def context_retrieve(query:str):
        """ 
        retrieve documents releated to the query from knowledge base
        
        
        """
        context = ""
        docs = vector_store.similarity_search(query=query,k=3)

        for doc in docs:
            context =  doc.page_content + "\n\n"

        return context

    system_prompt = """ you are helpfull assistant that answers question using retrieved context from the uploaded document 
        My knowledge base consistes of the details from the upload content.
        ALWAYS use the 'context_retrieve' tool for question requiring external knowledge   


    """

    agent = create_agent(
        model=llm_new,
        tools=[context_retrieve],
        system_prompt=system_prompt,
        checkpointer=memory   
          )
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# upload UI

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF Files" , type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):  
            path = "../Doc Files/" 
            for file in uploaded:
                with open(path+file.name , "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()
# chat ui

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anything related to uploaded documents...")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},config={"configurable":{"thread_id":2}})

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"AI","content":answer})