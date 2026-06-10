import sqlite3
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# Initialize Local LLM Engine
llm = ChatOllama(
    model="llama3.2", 
    temperature=0.7
)

# Define State Structure
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define LLM Node Process
def chat_node(state: ChatState) -> dict:
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Connect to Local SQLite Database (This file persists on your Mac's disk)
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build State Workflow Architecture
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# Compile into executable agent
chatbot = graph.compile(checkpointer=checkpointer)

def get_all_saved_threads():
    """Helper function to fetch historical threads directly from SQLite"""
    all_threads = set()
    try:
        # Pass an empty dictionary config mapping to list all threads in DB
        for checkpoint in checkpointer.list(config={}):
            t_id = checkpoint.config.get('configurable', {}).get('thread_id')
            if t_id:
                all_threads.add(t_id)
    except Exception:
        pass
    return list(all_threads)