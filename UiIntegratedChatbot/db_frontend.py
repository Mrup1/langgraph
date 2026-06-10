import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from db_backend import chatbot, get_all_saved_threads

def gen_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['thread_id'] = gen_thread_id()
    st.session_state['messages'] = []
    # Add the fresh ID to our tracker immediately so it appears on screen right away
    if st.session_state['thread_id'] not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(st.session_state['thread_id'])
    st.rerun()

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        if state and state.values and 'messages' in state.values:
            return state.values['messages']
    except Exception:
        pass
    return []

# 1. Initialize core system state trackers
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = gen_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# 2. Sync database records with our UI tracker on first launch
db_threads = get_all_saved_threads()
for t_id in db_threads:
    if t_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(t_id)

# Always make sure the active thread is part of the display list
if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(st.session_state['thread_id'])


# --- SIDEBAR UI ---
st.sidebar.title("Local Database Bot")

if st.sidebar.button("➕ New Chat Thread"):
    reset_chat()

st.sidebar.write("---")
st.sidebar.header("📜 Chat History Threads")

# Loop through our tracked thread list instead of hitting raw DB list directly
for thread_id in st.session_state['chat_threads'][::-1]:
    short_label = f"💬 Thread {thread_id[:8]}..."
    
    # Highlight the currently active communication stream path
    if thread_id == st.session_state['thread_id']:
        short_label = f"▶️ Active: {thread_id[:8]}..."
        
    if st.sidebar.button(short_label, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        raw_history = load_conversation(thread_id)
        
        # Unpack BaseMessage models into renderable dictionary objects
        ui_messages = []
        for msg in raw_history:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            ui_messages.append({'role': role, 'content': msg.content})
            
        st.session_state['messages'] = ui_messages
        st.rerun()


# --- MAIN CHAT SCREEN SPACE ---
st.caption(f"Connected to local SQLite storage bucket | Thread Context: `{st.session_state['thread_id']}`")

# Render historical text elements matching active window context
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input("Type your message here...")

if user_input and user_input.strip():
    # Append user entry block to interface log
    st.session_state['messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    # Stream text results token by token from local llama3.2 engine
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )
        
    st.session_state['messages'].append({"role": "assistant", "content": ai_message})
    st.rerun()