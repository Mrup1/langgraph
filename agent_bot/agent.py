import os
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from tools import (
    fetch_doctors,
    get_all_doctor_details,
    get_general_info,
    fetch_pricing,
    check_availability,
    book_appointment
)

# =========================================================
# 1. STATE DEFINITION
# =========================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agents: List[str]
    info_instructions: str
    booking_instructions: str
    loop_count: int


class ParallelRouterDecision(BaseModel):
    next_agents: List[Literal["InfoAgent", "BookingAgent"]] = Field(
        default_factory=list,
        description=(
            "Select ALL specialist nodes that need to run. "
            "If database data is already fetched and present in history, "
            "provide an empty list [] to trigger final synthesis."
        )
    )
    info_agent_instructions: str = Field(
        default="",
        description="Task/instructions for InfoAgent. Focus strictly on doctor directories and FAQs. Leave empty if not called."
    )
    booking_agent_instructions: str = Field(
        default="",
        description="Task/instructions for BookingAgent. Focus strictly on checking slots, pricing, and booking. Leave empty if not called."
    )
    reasoning: str = Field(description="Why these agents were chosen, or why synthesis is ready.")


# =========================================================
# 2. MODEL TIER INITIALIZATION
# =========================================================

# Tier 1: Router & worker LLM — deterministic, no streaming needed here
llm_fast = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Tier 2: Final synthesis LLM — streaming=True is REQUIRED so LangGraph can
# forward individual token chunks when stream_mode="messages" is used.
llm_smart = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    streaming=True          # ← must be True for token-level streaming
)


# =========================================================
# 3. NODE DEFINITIONS
# =========================================================

def supervisor_node(state: AgentState):
    """Tier 1 Router: Evaluates state and splits tasks."""
    loop_count = state.get("loop_count", 0)

    # Safeguard against infinite routing loops
    if loop_count >= 3:
        return {
            "next_agents": [],
            "info_instructions": "",
            "booking_instructions": "",
            "loop_count": loop_count + 1,
        }

    supervisor_prompt = """You are the Lead Medical Administrative Router.
Analyze the conversation history. Your primary job is to evaluate the user's request,
enforce strict domain boundaries, and delegate to specialist agents only when
hospital-related database extraction is required.

DOMAIN LOCK & GUARDRAILS (CRITICAL):
- You are strictly a hospital router. If a user asks for programming code (e.g., C++, Python),
  mathematical solutions, essays, or ANY topic unrelated to the hospital, YOU MUST REJECT IT.
- To reject: Return an empty list [] for `next_agents` to trigger the Synthesis node,
  and explicitly set instructions to politely refuse the non-hospital request.

DELEGATION & ROUTING:
If the request is valid and requires new data, delegate to the appropriate specialist agents
by returning their names in `next_agents`:
1. `InfoAgent`: For doctor directories, finding doctors, and FAQs/general hospital info.
2. `BookingAgent`: For checking availability/slots, consulting services pricing, and booking.

CRITICAL EXECUTION RULES:
- MULTI-INTENT: If the user has a multi-intent query (e.g., asking for doctor directory AND
  pricing/slots), select BOTH agents ["InfoAgent", "BookingAgent"] so they run in PARALLEL.
- SPECIFICITY: Carefully extract and write isolated, specific instruction strings for
  `info_agent_instructions` and `booking_agent_instructions`.
- AVOID DUPLICATION: If the requested data has already been fetched by the tools and is present
  in the history, return an empty list [] in `next_agents` to trigger the final Synthesis node.
- NO ASSUMED BOOKINGS: DO NOT instruct the BookingAgent to book an appointment unless the user
  explicitly requested a booking. Checking availability or pricing is NOT a booking request.
- ABSOLUTE CONFIDENCE: When routing to Synthesis (returning []), ensure your instructions dictate
  that the final response must present all schedules, pricing, and details as absolute, confirmed
  facts without weak disclaimers like "subject to change."
"""

    messages = [{"role": "system", "content": supervisor_prompt}] + state["messages"]

    structured_llm = llm_fast.with_structured_output(ParallelRouterDecision)
    decision = structured_llm.invoke(messages)

    next_agents = decision.next_agents if decision.next_agents else []

    return {
        "next_agents": next_agents,
        "info_instructions": decision.info_agent_instructions,
        "booking_instructions": decision.booking_agent_instructions,
        "loop_count": loop_count + 1,
    }


def _safe_content(content) -> str:
    """Normalise tool-message content which can be str or list[dict]."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(content)


def info_agent_node(state: AgentState):
    """Tier 1 Worker: Fetches doctor/FAQ data and executes tools internally."""
    instructions = state.get("info_instructions", "")
    if not instructions:
        return {"messages": []}

    system_prompt = f"""You are the InfoAgent, a specialised database retrieval assistant for
doctor directories and FAQs.
Your current task: {instructions}

CRITICAL GUIDELINES:
1. You ONLY have access to: fetch_doctors, get_all_doctor_details, get_general_info.
2. Your sole job is to call these tools to retrieve the data requested in your task.
3. DO NOT write a conversational response to the patient. Do not explain what you cannot do.
"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    info_tools = [fetch_doctors, get_all_doctor_details, get_general_info]
    llm_with_tools = llm_fast.bind_tools(info_tools)
    response = llm_with_tools.invoke(messages)
    response.name = "InfoAgent"

    outputs = [response]

    if response.tool_calls:
        tool_map = {
            "fetch_doctors": fetch_doctors,
            "get_all_doctor_details": get_all_doctor_details,
            "get_general_info": get_general_info,
        }
        for tc in response.tool_calls:
            if tc["name"] in tool_map:
                try:
                    res = tool_map[tc["name"]].invoke(tc["args"])
                    outputs.append(
                        ToolMessage(
                            content=str(res),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
                except Exception as exc:
                    outputs.append(
                        ToolMessage(
                            content=f"Error executing tool {tc['name']}: {exc}",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )

    return {"messages": outputs}


def booking_agent_node(state: AgentState):
    """Tier 1 Worker: Fetches slots/prices and executes tools internally."""
    instructions = state.get("booking_instructions", "")
    if not instructions:
        return {"messages": []}

    system_prompt = f"""You are the BookingAgent, a specialised database retrieval assistant for
pricing, checking slots, and booking appointments.
Your current task: {instructions}

CRITICAL GUIDELINES:
1. You ONLY have access to: fetch_pricing, check_availability, book_appointment.
2. Your sole job is to call these tools to retrieve the data requested in your task.
3. DO NOT write a conversational response to the patient. Do not explain what you cannot do.
"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    booking_tools = [fetch_pricing, check_availability, book_appointment]
    llm_with_tools = llm_fast.bind_tools(booking_tools)
    response = llm_with_tools.invoke(messages)
    response.name = "BookingAgent"

    outputs = [response]

    if response.tool_calls:
        tool_map = {
            "fetch_pricing": fetch_pricing,
            "check_availability": check_availability,
            "book_appointment": book_appointment,
        }
        for tc in response.tool_calls:
            if tc["name"] in tool_map:
                try:
                    res = tool_map[tc["name"]].invoke(tc["args"])
                    outputs.append(
                        ToolMessage(
                            content=str(res),
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )
                except Exception as exc:
                    outputs.append(
                        ToolMessage(
                            content=f"Error executing tool {tc['name']}: {exc}",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                    )

    return {"messages": outputs}


def synthesis_node(state: AgentState):
    """Tier 2 Writer: Compiles the final human-facing response (streaming-enabled).

    IMPORTANT: We call llm_smart.invoke() here — NOT .stream().
    LangGraph's stream_mode="messages" intercepts the model's token stream
    automatically when streaming=True is set on the ChatGroq instance,
    so the Streamlit UI receives chunks without any extra plumbing in this node.
    """
    system_prompt = """You are the Lead Medical Administrative Assistant.
The database specialist agents have gathered all necessary raw data in the conversation history.
Read the conversation history and raw tool outputs, then synthesise a polite, professional,
clear, and comprehensive final response for the patient.

Guidelines:
- Never mention 'agents', 'tools', 'nodes', or database names to the user.
- Format lists, tables, or times cleanly and professionally.
- Be empathetic, welcoming, and precise.
- Provide a complete answer to ALL parts of the user's query.
- State schedules, pricing, and availability as confirmed facts — avoid weak qualifiers.
"""

    # Build a clean message list: system prompt + full conversation history.
    # We exclude any AIMessage that has tool_calls to avoid confusing the synthesis LLM
    # (it only needs the tool *results*, not the intermediate tool-calling steps).
    filtered_messages = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Keep the tool results that follow; skip the intermediate AI tool-call message
            continue
        filtered_messages.append(msg)

    messages = [SystemMessage(content=system_prompt)] + filtered_messages

    # invoke() triggers streaming under the hood when streaming=True on the model.
    # LangGraph will emit each token chunk to the stream_mode="messages" consumer.
    response = llm_smart.invoke(messages)
    return {"messages": [response]}


# =========================================================
# 4. CONDITIONAL ROUTING LOGIC
# =========================================================

def parallel_router_edge(state: AgentState):
    next_agents = state.get("next_agents", [])
    if not next_agents:
        return "Synthesis"
    # Return a list so LangGraph fans out to all selected agents in parallel
    return next_agents


# =========================================================
# 5. GRAPH COMPILATION
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("InfoAgent", info_agent_node)
workflow.add_node("BookingAgent", booking_agent_node)
workflow.add_node("Synthesis", synthesis_node)

workflow.add_edge(START, "Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    parallel_router_edge,
    {
        "InfoAgent": "InfoAgent",
        "BookingAgent": "BookingAgent",
        "Synthesis": "Synthesis",
    },
)

# Workers feed back into Supervisor for the second-pass routing decision
workflow.add_edge("InfoAgent", "Supervisor")
workflow.add_edge("BookingAgent", "Supervisor")

# Synthesis is terminal
workflow.add_edge("Synthesis", END)

memory = MemorySaver()
hospital_agent = workflow.compile(checkpointer=memory)