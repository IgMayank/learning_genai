from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper



llm = ChatGroq(model="openai/gpt-oss-20b",streaming=True)
search = GoogleSerperAPIWrapper()
tool =[search.run]
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history=[]



agent = create_agent(
    model = llm,
    tools = tool,
    checkpointer=st.session_state.memory,
    system_prompt="you are an amazing agent and can search on google as well ",
)


st.subheader("Quickchat-Speed Of Light")
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)



query = st.chat_input("Ask Anything")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})
    response = agent.stream(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":357}},
        stream_mode="messages"
)
    

    ai_container =st.chat_message("ai")
    # we will create a refernce here
    with ai_container:
        space = st.empty()

        mess=""
        for chunk in response:
            mess = mess+chunk[0].content
            space.write(mess)





    
    st.session_state.history.append({"role":"ai","content":mess})