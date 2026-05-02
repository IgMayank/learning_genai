from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain_community.utilities import SQLDatabase 
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI

import streamlit as st



st.subheader("🤖🗂️ TaskBot — Your Personal Task Manager")
if "messages" not in st.session_state:
    st.session_state.messages=[]


for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])
    




db = SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
    CREATE TABLE IF NOT EXISTS tasks(
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       title TEXT NOT NULL,
       description TEXT,
       status TEXT CHECK(status IN ('Pending','In_Progress','Done'))DEFAULT 'Pending',
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)


system_prompt= """
You are a task management assistant that interacts  with sql  databse named 'my_tasks' t

follow these rules:
1. Limit SELECT queries with 40 query max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE , Confirm with SELECT query 
3. If the user requests a list of tasks , present the output in a structured table format to ensure a clean and organized display in the browser."
4. Follow the structure strictly and use stack based listing for eg if a task is listed at id no1 make sure it stays at the top and the upcoming ids such as 2,3 come after  id 1.
5. if user ask to show database , show him right away regardless if the db is empty.

 CRUD OPERATIONS:
             
             CREATE : INSERT into my_tasks(title,description,status)
             READ : SELECT * FROM my_tasks  WHERE... Limit 10
             UPDATE : UPDATE my_tasks SET STATUS=? WHERE ID =? OR TITLE=?
             DELETE : DELETE FROM my_tasks WHERE ID=? OR TITLE=?


Table Schema: Id , Title ,Description,Status(Pending,In_progress,Done),created_at.
"""



model =ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
search = GoogleSerperAPIWrapper()
toolkit = SQLDatabaseToolkit(db=db,llm=model)
tool = toolkit.get_tools()




@st.cache_resource # this is used so the memory is not created again and again cause when we use stream lit it refreshes page agin and again
def get_agent():
    agent = create_agent( 
        model=model,
        tools=tool,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt ,
    )
    return agent 

agent = get_agent()



prompt = st.chat_input("Lets Manage Together")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message('ai'):
        with st.spinner("processing... "):
            response = agent.invoke(
                {"messages":[{"role":"user","content":prompt}]},
                config={"configurable":{"thread_id":"111"}}
            )
            result = response["messages"][-1].content
            st.markdown(result)
            st.session_state.messages.append({"role":"ai","content":result})
            