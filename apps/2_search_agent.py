from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent 
from langgraph.checkpoint.memory import MemorySaver


llm = ChatGroq(model="openai/gpt-oss-20b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model = llm,
    tools=[search.run],
    system_prompt="you are an agent who can search on google for any question",
    checkpointer=MemorySaver()
)

while True :
    query=input("Enter Your Question: ")
    if query.lower() == quit:
        print("Ai: Good Bye Sir")
        break
    
    response = agent.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"test123"}}
        )
    print("AI: ",response["messages"][-1].content)