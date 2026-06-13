import os
import uuid
from dotenv import load_dotenv

# 1. Load the .env file BEFORE importing the agent
load_dotenv()

# 2. Import the agent
from agent import hospital_agent

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
    print("Hospital Assistant Session Initialized.")
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

            # Stream intermediate node activations step-by-step
            events = hospital_agent.stream(
                {"messages": [("user", user_input)]}, 
                config, 
                stream_mode="updates"
            )
            
            last_message = None
            for event in events:
                for node_name, node_output in event.items():
                    print(f"\n[Node: {node_name}]")
                    
                    # Print node specific updates
                    if "next_agents" in node_output and node_output["next_agents"]:
                        print(f"  -> Routing to: {node_output['next_agents']}")
                    if "info_instructions" in node_output and node_output["info_instructions"]:
                        print(f"  -> Info Instructions: {node_output['info_instructions']}")
                    if "booking_instructions" in node_output and node_output["booking_instructions"]:
                        print(f"  -> Booking Instructions: {node_output['booking_instructions']}")
                        
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            last_message = msg
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                print(f"  -> Tool Calls: {[tc['name'] for tc in msg.tool_calls]}")
                            elif msg.type == "tool":
                                # Tool response
                                print(f"  -> Tool Response: {msg.content[:200]}...")
                            elif node_name != "Synthesis":
                                # Intermediate AI messages
                                print(f"  -> State Message: {msg.content[:100]}...")

            if last_message and last_message.content:
                print(f"\nAssistant Response:\n{last_message.content}")

        except KeyboardInterrupt:
            print("\nSession aborted safely.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    run_interaction_session()