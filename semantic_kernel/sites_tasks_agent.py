import asyncio
import json
import logging
import os
from typing import Any, Dict

from semantic_kernel.agents import HandoffOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import (AuthorRole, ChatMessageContent,
                                      FunctionCallContent,
                                      FunctionResultContent)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.logging import setup_logging

from dotenv import load_dotenv;load_dotenv()


# Global variable to hold the loaded data, as plugins need access to it.
# In a more robust system, this might be managed via dependency injection or a service layer.
all_data: Dict[str, Any] = {}

setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)

from core.agents import get_agents
from core.plugins import SiteTasksPlugin


def agent_response_callback(message: ChatMessageContent) -> None:
    """Observer function to print the messages from the agents."""
    print(f"\n--- Agent Response ({message.name}) ---")
    if message.content:
        print(f"Content: {message.content}")
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"Function Call: '{item.name}' with arguments '{item.arguments}'")
        if isinstance(item, FunctionResultContent):
            # For large JSON results, truncate for display in the console
            result_str = str(item.result)
            if len(result_str) > 500:
                print(f"Function Result from '{item.name}':\n{result_str[:500]}...\n(Truncated for display)")
            else:
                print(f"Function Result from '{item.name}':\n{result_str}")
    print("-------------------------------------")


async def human_response_function() -> ChatMessageContent:
    """Function to get human input when an agent requests it (e.g., for clarification)."""
    user_input = input("Agent requires input. User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


async def run_semantic_kernel_agent_query(user_query: str, context_json_path: str) -> str:
    """
    Main function to run the Semantic Kernel handoff orchestration.
    """
    global all_data 

    # Load JSON data from file (only once if not already loaded)
    if not all_data: # Load only if empty
        try:
            with open(context_json_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            print(f"Successfully loaded data from {context_json_path}")
        except FileNotFoundError:
            print(f"Error: The file '{context_json_path}' was not found.")
            return "Error: Data file not found."
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{context_json_path}'. Check file format.")
            return "Error: Invalid JSON data."
        except Exception as e:
            print(f"An unexpected error occurred while loading data: {e}")
            return "Error: Data loading failed."

    # Initialize the Kernel
    kernel = Kernel()

    # Create the plugin instance with the loaded data
    data_plugin_instance = SiteTasksPlugin(all_data)

    # 1. Create agents and define handoff relationships
    agents, handoffs = get_agents(kernel, data_plugin_instance)
    handoff_orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
    )

    # 2. Create a runtime and start it
    runtime = InProcessRuntime() 
    runtime.start()

    print(f"\n--- Initiating Semantic Kernel with query: '{user_query}' ---")
    
    # 3. Invoke the orchestration with the user's query
    orchestration_result = await handoff_orchestration.invoke(
        task=user_query,
        runtime=runtime,
    )

    # 4. Wait for the results
    final_result_content = ""
    value = await orchestration_result.get()
    if isinstance(value, ChatMessageContent):
        final_result_content = value.content
    elif isinstance(value, str):
        final_result_content = value
    else:
        final_result_content = str(value) # Fallback for other content types
    
    print(f"\n--- Final Orchestration Result ---")
    print(final_result_content)

    # 5. Stop the runtime after the invocation is complete
    await runtime.stop_when_idle()
    print("Runtime stopped.")
    
    return final_result_content


async def main():
    """
    Main function to run the interactive agent system.
    """
    # Create a dummy data.json for testing if it doesn't exist
    data_json_path = 'knowledge/data.json'

    print("\n--- Welcome to the Project Assistant ---")
    print("Type your queries about sites and tasks. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Exiting Project Assistant. Goodbye!")
            break
        
        await run_semantic_kernel_agent_query(user_input, data_json_path)


if __name__ == "__main__":
    asyncio.run(main())