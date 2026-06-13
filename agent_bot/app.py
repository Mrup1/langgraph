import streamlit as st
import uuid
import os
from dotenv import load_dotenv

# CRITICAL: Load environment variables BEFORE importing the agent
load_dotenv()

# Initialize Database safely
if not os.path.exists("hospital.db"):
    import database
    database.init_db()

# Now it is safe to import the agent because the environment variables are active
from agent import hospital_agent

# Configure Streamlit Page
st.set_page_config(page_title="Hospital AI Assistant", page_icon="🏥", layout="centered")
st.title("🏥 Hospital AI Assistant")
st.markdown("Ask complex multi-intent questions like: *'Who are your cardiologists, what do they charge, and are they available on 2026-06-15?'*")
st.markdown("---")

# Session State Management for conversational memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History on Rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle User Input
if user_input := st.chat_input("Ask a question or book an appointment..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        inputs = {"messages": [("user", user_input)]}
        
        with st.spinner("Accessing hospital records..."):
            final_text = ""
            try:
                events = hospital_agent.stream(inputs, config, stream_mode="values")
                for event in events:
                    if "messages" in event:
                        last_msg = event["messages"][-1]
                        if last_msg.type == "ai" and last_msg.content:
                            final_text = last_msg.content
                            response_placeholder.markdown(final_text)

                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")