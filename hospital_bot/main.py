import os
import uuid
from dotenv import load_dotenv
from agent import hospital_agent

load_dotenv()
def run_interaction_session():
    # Verify database state footprint
    if not os.path.exists("hospital.db"):
        print("Warning: hospital.db not found. Running database initializer...")
        import database
        database.init_db()

    # Generate isolated conversation session channel identifier
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 70)
    print(f"Hospital Assistant Session Initialized.")
    print(f"Active Memory Thread ID: {thread_id}")
    print("=" * 70)
    print("Ask complex multi-intent questions like:")
    print("'Who are your cardiologists, what do they charge, and are they available on 2026-06-15?'")
    print("-" * 70)

    while True:
        try:
            user_input = input("\nPatient Query: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Session terminated safely.")
                break
                
            if not user_input:
                continue

            # Stream intermediate tool activations step-by-step
            events = hospital_agent.stream(
                {"messages": [("user", user_input)]}, 
                config, 
                stream_mode="values"
            )
            
            last_message = None
            for event in events:
                if "messages" in event:
                    last_message = event["messages"][-1]
            
            if last_message:
                print(f"\nAssistant Response:\n{last_message.content}")

        except KeyboardInterrupt:
            print("\nSession aborted safely.")
            break

if __name__ == "__main__":
    # Ensure standard key validation is satisfied before boot execution traces
    if "OPENAI_API_KEY" not in os.environ:
        # Prompt developer input if variable isn't injected into system environment
        os.environ["OPENAI_API_KEY"] = input("Please input your OpenAI API Key to test run: ").strip()
        
    run_interaction_session()