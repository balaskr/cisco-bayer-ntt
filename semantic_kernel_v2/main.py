from core.agents.manager_agent import ManagerAgent
import asyncio


async def main():
    manager_agent_instance = ManagerAgent()

    """
    Main function to run the interactive agent system locally.
    """
    print("\n--- Welcome to the Project Assistant (Local) ---")
    print("Type your queries about sites and tasks. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Exiting Project Assistant. Goodbye!")
            break
        
        response = await manager_agent_instance.run(user_input)
        print(response)

        
if __name__ == "__main__":
    asyncio.run(main())