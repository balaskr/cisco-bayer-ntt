# core/agents/manager_agent.py
import os
import asyncio

# Core AgentBuilder
from core.agents.agent_builder import AgentBuilder

# Plugins
from core.plugins import DelegationPlugin # Import the plugin where DelegationPlugin is defined

# LLM Service (assuming this setup is common and can be imported or passed in)
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion

# Global LLM service instance (reusing the common pattern from sites_tasks_agent)
_manager_chat_completion_service = GoogleAIChatCompletion(
    gemini_model_id="gemini-1.5-flash", # Could use a different model for manager if desired
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Instructions for the Manager Agent
manager_orchestrator_instructions = (
    """You are a top-level AI Project Orchestrator. Your primary role is to understand complex user queries and
    determine the most appropriate specialized agent or tool to handle them.

    **Available Tools (Plugins):**
    - `call_sites_tasks_agent(query: str)`: Use this tool to delegate any query that is specifically about **site details, searching sites, task details, or tasks associated with a specific site**. This agent is highly specialized in data retrieval and summarization concerning project sites and tasks.

    **General Instructions & Response Formatting:**
    - **Delegation First:** Always consider if a query can be fully answered by delegating to a specialized agent. If a tool exists for the user's intent, use it.
    - **Clarity & Conciseness:** Provide clear, concise, and easy-to-understand responses.
    - **Markdown Output:** All your responses should be formatted in Markdown.
    - **Clarification:** If the user's request is ambiguous, unclear, or you cannot fulfill it with your tools, ask for clarification.
    - **Error Handling:** If a delegated tool returns an error, inform the user appropriately and suggest alternative phrasing or actions.
    - **General Conversation:** For queries that are not about sites or tasks (or other future specialized domains), respond conversationally and politely.
    -**if asked what is your purpose or who are you, explain concisely about your role and functionality**
    """
)

class ManagerAgent(AgentBuilder):
    def __init__(self):
        # Instantiate the DelegationPlugin
        delegation_plugin_instance = DelegationPlugin()

        super().__init__(
            name="ProjectManagerAgent",
            description="Orchestrates various specialized agents to answer complex project-related queries.",
            instructions=manager_orchestrator_instructions,
            service=_manager_chat_completion_service,
            plugins=[delegation_plugin_instance] # Register the delegation plugin
        )


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