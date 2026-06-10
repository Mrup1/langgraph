import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
import uuid 

def gen_thread_id():
    # Convert to string format so it handles cleanly across dictionaries
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['thread_id'] = gen_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['messages'] = []
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    # FIX 1: Safely handle instances where the thread has zero history saved yet
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        if state and state.values and 'messages' in state.values:
            return state.values['messages']
    except Exception:
        pass
    return []

# Initialize session structures
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = gen_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

# --- SIDEBAR INTERFACE ---
st.sidebar.title("Chatbot Operations")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My History Threads")

for thread_id in st.session_state['chat_threads'][::-1]:
    # Render short, human-readable tags for the UUID strings in the sidebar
    short_label = f"💬 Thread {thread_id[:8]}..."
    
    # FIX 2: Added a unique structural key to every button generated inside the loop
    if st.sidebar.button(short_label, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})
            
        st.session_state['messages'] = temp_messages
        st.rerun()

# --- MAIN SCREEN INTERFACE ---
st.subheader(f"Active Thread ID: {st.session_state['thread_id']}")
st.write("---")

# Render active window messages logs
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

user_input = st.chat_input("Type your message here...")

if user_input and user_input.strip():
    # Append user input message to local UI state array and render it
    st.session_state['messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Establish localized configuration pointers
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
    
    # FIX 3: REMOVED the extra duplicate chatbot.invoke() call. 
    # Having both invoke and stream running back-to-back doubles execution work and corrupts context logs.
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,  # FIX 4: Swapped out hardcoded 'thread-1' for active UUID pointer
                stream_mode='messages'
            )
        )
        
    st.session_state['messages'].append({"role": "assistant", "content": ai_message})
    st.rerun()