import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type your message here...")

# FIX: Strip whitespace and explicitly verify user_input has text content
if user_input and user_input.strip():
    # 1. Append user message to local UI state and render it
    st.session_state['messages'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    # 2. Invoke LangGraph safely now that content is guaranteed to be a valid string
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": "1"}}
    )
    
    # 3. Extract AI response content from state list
    ai_message = response['messages'][-1].content

    # 4. Append AI message to UI state and render it
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config={'configurable':{'thread_id':'thread-1'}},
                stream_mode='messages'
            )
        )
    st.session_state['messages'].append({"role": "assistant", "content": ai_message})
    # Force a rerun to show the newly added assistant response cleanly
    st.rerun()