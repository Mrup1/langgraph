import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
# 1. IMPORT CHATGROQ INSTEAD OF OLLAMA
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from tools import hospital_tools

load_dotenv()

# --- State Topology Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# --- System Prompt Definition ---
SYSTEM_PROMPT = """You are an expert, highly intelligent medical administrative assistant for the hospital.

You have access to specific local database tools to extract and write real-time information:
1. `fetch_doctors`: Query staff listings by specialization fields.
2. `fetch_pricing`: Find billing, consultation fees, and procedure costs.
3. `check_availability`: Scan active schedules to determine free slots for a given date (YYYY-MM-DD).
4. `book_appointment`: Register structured appointment reservations.
5. `get_general_info`: Resolve FAQs regarding location, emergency services, or policy workflows.

MULTI-INTENT CAPABILITY INSTRUCTION:
Users often bundle multiple operational requests into a single message. You must address every aspect of their query in parallel or sequential tool chains during a single turn.
Example: "Who is your best cardiologist, what do they charge, and are they free tomorrow?"
Always output valid tool call structures. If the user asks for a date, assume today is June 10, 2026.
"""


llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0
)

# Bind the hospital tools to the Groq LLM
llm_with_tools = llm.bind_tools(hospital_tools)

# --- Graph Node Formations ---
def chatbot_node(state: AgentState):
    """Executes the LLM logic layer to determine tool invocation requirements."""
    messages = [({"role": "system", "content": SYSTEM_PROMPT})] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- Graph Assembly Line ---
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(hospital_tools))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

memory = MemorySaver()
hospital_agent = workflow.compile(checkpointer=memory)