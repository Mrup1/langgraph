import os
import uuid
import streamlit as st
from dotenv import load_dotenv

# ── 1. Load env vars BEFORE any project imports ──────────────────────────────
load_dotenv()

# ── 2. Database bootstrap (only once) ────────────────────────────────────────
if not os.path.exists("hospital.db"):
    import database
    database.init_db()

# ── 3. Import agent (env + DB are ready) ─────────────────────────────────────
from agent import hospital_agent

# ── 4. Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hospital AI Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Hospital AI Assistant")
st.caption(
    "Ask complex questions like: *'Who are your cardiologists, what do they charge, "
    "and are they available on 2026-06-15?'*"
)
st.divider()

# ── 5. Session state ──────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": str, "content": str}]

# ── 6. Render existing chat history ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 7. Handle new user input ──────────────────────────────────────────────────
if user_input := st.chat_input("Ask a question or book an appointment..."):

    # Persist + display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── All assistant output lives inside ONE chat bubble ─────────────────────
    with st.chat_message("assistant"):
        # Single status widget: shows spinner during routing, collapses when done
        status = st.status("Accessing hospital records…", expanded=False)

        # The streaming text lives here — outside the status widget
        response_placeholder = st.empty()

    # ── Stream the graph ──────────────────────────────────────────────────────
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": [("user", user_input)]}
    final_text = ""

    try:
        for chunk, metadata in hospital_agent.stream(
            inputs, config, stream_mode="messages"
        ):
            node = metadata.get("langgraph_node", "")

            # Update status label to show which agent is active
            if node == "Supervisor":
                status.update(label="🔀 Routing your request…")
            elif node == "InfoAgent":
                status.update(label="🔍 Fetching doctor information…")
            elif node == "BookingAgent":
                status.update(label="📅 Checking availability & pricing…")
            elif node == "Synthesis":
                # First synthesis chunk — mark routing as done
                if not final_text:
                    status.update(label="✅ Done", state="complete", expanded=False)

                # Extract token text safely (content can be str or list)
                content = chunk.content
                if isinstance(content, str):
                    token = content
                elif isinstance(content, list):
                    token = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                else:
                    token = str(content) if content else ""

                if token:
                    final_text += token
                    response_placeholder.markdown(final_text + "▌")

        # Final render — remove the blinking cursor
        if final_text:
            response_placeholder.markdown(final_text)
        else:
            final_text = "I'm sorry, I couldn't generate a response. Please try again."
            response_placeholder.markdown(final_text)
            status.update(label="⚠️ No response generated", state="error")

    except Exception as exc:
        status.update(label=f"❌ Error", state="error")
        final_text = f"An error occurred: {exc}"
        response_placeholder.error(final_text)

    # Persist assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": final_text})