import asyncio
import json
import logging
import os
from typing import Any, Dict, List

# Semantic Kernel Imports
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.google.google_ai import \
    GoogleAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import (AuthorRole, ChatMessageContent,
                                      FunctionCallContent,
                                      FunctionResultContent)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.logging import setup_logging

from core.plugins import SiteTasksPlugin

from dotenv import load_dotenv;load_dotenv()


chat_completion_service = GoogleAIChatCompletion(
    gemini_model_id="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)


# Global variable to hold the loaded data.
# This will be populated from the local data.json file.
_global_site_tasks_data: Dict[str, Any] = {}

# Configure Semantic Kernel logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)


async def handle_intermediate_steps(msg: ChatMessageContent) -> None:
    if any(isinstance(item, FunctionResultContent) for item in msg.items):
        for fr in msg.items:
            if isinstance(fr, FunctionResultContent):
                print(f"Function Result:> {fr.result} for function: {fr.name}")
    elif any(isinstance(item, FunctionCallContent) for item in msg.items):
        for fcc in msg.items:
            if isinstance(fcc, FunctionCallContent):
                print(f"Function Call:> {fcc.name} with arguments: {fcc.arguments}")

# --- Orchestrator Agent Definition ---

def get_orchestrator_agent(sk_kernel: Kernel, data_plugin: SiteTasksPlugin) -> ChatCompletionAgent:
    """
    Returns a single orchestrator agent responsible for all queries using its registered plugins.
    """
    # llm_service = AzureChatCompletion(
    #     service_id="default",
    #     endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY")
    # )
    # sk_kernel.add_service(llm_service)

    orchestrator_instructions = (
        """You are an advanced AI Project Assistant. Your primary role is to understand user queries about sites and tasks, 
        and use the provided tools (plugins) to retrieve or synthesize the requested information. 
        You are the sole decision-maker for which tool to use, if any, and how to format the final response.

        **Available Tools (Plugins):**
        - `get_site_details(site_query: str)`: Use this when the user asks for **details of a specific site** (e.g., "Show me site ATH2", "Details for Maroussi?").
        - `search_sites(query: str)`: Use this when the user asks to **list sites based on attributes** or a partial name (e.g., "List all sites in Bangalore", "Show me sites with status Active"). This can also be used to "list all sites" by providing a broad query like "all" or "active".
        - `get_task_details(task_query: str)`: Use this when the user asks for **details of a specific task** (e.g., "What is task T1?", "Show me progress of task 123").
        - `get_tasks_for_site(site_query: str)`: Use this when the user asks for **tasks associated with a specific site** (e.g., "tasks for site ATH2", "What tasks are at Bangalore Tech Park?").
        - `get_all_data_json()`: Use this when the user asks for **overall/aggregate information across the dataset**, an **executive summary**, or to **list all data/sites** for analysis that requires the entire dataset. You will then process this JSON yourself.

        **General Instructions & Response Formatting:**
        - **Accuracy:** Always strive for factual accuracy based on the data retrieved from your tools.
        - **Clarity:** Provide clear, concise, and easy-to-understand responses.
        - **Markdown Output:** All your responses should be formatted in Markdown. Never return raw JSON or code blocks in your final response to the user, unless explicitly asked to show the raw data.
        - **Site Details Formatting (for `get_site_details` results):** Present information as a list:
            `- location_name: [name]`
            `- site_id: [id]`
            `- status: [status value from 'state']`
            `- Phase: [value from 'latitude']`
            `- Company Size: [value from 'address_2']`
        - **Site List Formatting (for `search_sites` results or 'list all'):** Provide a formatted Markdown list.
        - **Task Details Formatting (for `get_task_details` results):** Extract key information from the returned JSON and present it in a readable Markdown list or paragraph.
        - **Tasks for Site Formatting (for `get_tasks_for_site` results):** Provide a formatted Markdown list of tasks.
        - **Overall/Summary/List All (using `get_all_data_json`):**
            - **Overall/Aggregate:** After calling `get_all_data_json`, parse the JSON, perform the requested aggregation (e.g., count active sites, total tasks), and present the numerical result clearly.
            - **Summary:** After calling `get_all_data_json`, analyze the JSON to provide a high-level executive summary, including: a short description of the data, task details summary, site states, task statuses, total sites, and total tasks (with their statuses). Remember to relabel "latitude" as "Phase" and "address_2" as "Company Size" in your summary if referring to those concepts.
            - **List All Sites:** After calling `get_all_data_json`, iterate through all sites and list them with their 'location_name', 'site_id', and 'status'.
        - **Clarification:** If the user's request is ambiguous, unclear, or you cannot fulfill it with your tools or current data, ask for clarification by politely stating: "I need more information to help you. Could you please specify your request?"
        - **Error Handling:** If a tool returns an error or no data, inform the user appropriately.
        """
    )

    orchestrator_agent = ChatCompletionAgent(
        name="ProjectAssistantOrchestrator",
        description="A comprehensive AI assistant for project site and task management.",
        instructions=orchestrator_instructions,
        service=chat_completion_service,
        plugins=[data_plugin] # Register all plugin functionalities with this single orchestrator
    )
    return orchestrator_agent

# --- Main Query Runner ---
async def run_semantic_kernel_agent_query(user_query: str) -> str:
    """
    Loads data from a local JSON file and then runs the Semantic Kernel single orchestrator agent.
    """
    global _global_site_tasks_data 

    # 1. Load data from the local JSON file (only if not already loaded)
    data_json_path = 'knowledge/data.json' # Adjusted path for common local setup
    if not _global_site_tasks_data: 
        try:
            with open(data_json_path, 'r', encoding='utf-8') as f:
                _global_site_tasks_data = json.load(f)
            logging.info(f"Successfully loaded data from {data_json_path}")
        except FileNotFoundError:
            logging.error(f"Error: The file '{data_json_path}' was not found.")
            return "Error: Data file not found."
        except json.JSONDecodeError:
            logging.error(f"Error: Could not decode JSON from '{data_json_path}'. Check file format.")
            return "Error: Invalid JSON data."
        except Exception as e:
            logging.exception("An unexpected error occurred while loading data.")
            return f"Error: An unexpected error occurred during data loading: {str(e)}."

    # Initialize the Kernel
    kernel = Kernel()

    # Create the plugin instance with the globally loaded data
    data_plugin_instance = SiteTasksPlugin(_global_site_tasks_data)

    # Get the single orchestrator agent
    orchestrator_agent = get_orchestrator_agent(kernel, data_plugin_instance)

    async for response in orchestrator_agent.invoke(
        messages=[ChatMessageContent(role=AuthorRole.USER, content=user_query)],
        on_intermediate_message=handle_intermediate_steps
    ):
        print(f"# {response.name}: {response}")
    

# --- Interactive Main Execution for Local Testing ---
async def main():
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
        
        await run_semantic_kernel_agent_query(user_input)
        


if __name__ == "__main__":
    asyncio.run(main())

