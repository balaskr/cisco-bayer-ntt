import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from semantic_kernel.agents import (Agent, ChatCompletionAgent,
                                    HandoffOrchestration,
                                    OrchestrationHandoffs)
from semantic_kernel.agents.runtime import InProcessRuntime
# Using AzureChatCompletion for Azure OpenAI.
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import (AuthorRole, ChatMessageContent,
                                      FunctionCallContent,
                                      FunctionResultContent)
from core.utils import search_json_objects
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.logging import setup_logging

from dotenv import load_dotenv;load_dotenv()


# Global variable to hold the loaded data, as plugins need access to it.
# In a more robust system, this might be managed via dependency injection or a service layer.
all_data: Dict[str, Any] = {}

setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)


# --- Plugin Definitions ---

class SiteTasksPlugin:
    """
    A plugin to interact with site and task data.
    The entire dataset is passed to its constructor.
    """
    _all_data: Dict[str, Any]

    def __init__(self, all_data_param: Dict[str, Any]):
        self._all_data = all_data_param # Assign the passed data to the instance

    @kernel_function(
        name="get_site_details",
        description="Given a site ID or name, retrieve detailed information for that single site. Returns site details in Markdown.",
    )
    def get_site_details(self, site_query: str) -> str:
        """
        Retrieves details for a specific site from the loaded data.
        Assumes site_query is a specific ID or name for a single site.
        """
        sites = self._all_data.get("data", [])
        filtered_sites = search_json_objects(sites, site_query)

        if not filtered_sites:
            return f"No site found matching '{site_query}'. Please clarify the site ID or name."
        
        # Prioritize exact ID match or the first most relevant result
        site_found = None
        for site in filtered_sites:
            if site.get("site_id", "").lower() == site_query.lower() or site.get("location_name", "").lower() == site_query.lower():
                site_found = site
                break
        if not site_found and filtered_sites: # If no exact match, just take the first result
            site_found = filtered_sites[0]


        if site_found:
            return (
                f"- location_name: {site_found.get('location_name', 'N/A')}\n"
                f"- site_id: {site_found.get('site_id', 'N/A')}\n"
                f"- status: {site_found.get('state', 'N/A')}\n"
                f"- Phase: {site_found.get('latitude', 'N/A')}\n" # Placeholder for Phase from CrewAI, maps to 'latitude'
                f"- Company Size: {site_found.get('address_2', 'N/A')}" # Placeholder for Company Size from CrewAI, maps to 'address_2'
            )
        return f"No relevant site found for '{site_query}' within the provided data."

    @kernel_function(
        name="search_sites",
        description="Searches and lists multiple sites based on a search query, like 'sites in Bangalore' or 'active sites'. Returns a Markdown list of site names, IDs, and statuses.",
    )
    def search_sites(self, query: str) -> str:
        """
        Searches for sites based on a general query and returns a formatted list.
        """
        print("here")
        sites = self._all_data.get("data", [])

        filtered_sites = search_json_objects(sites, query)
        print(filtered_sites)

        if not filtered_sites:
            return f"No sites found matching the criteria: '{query}'."

        output = "### Found Sites:\n\n"
        for site in filtered_sites:
            output += (
                f"- **{site.get('location_name', 'N/A')}** (ID: {site.get('site_id', 'N/A')}) - Status: {site.get('state', 'N/A')}\n"
            )
        print(output)
        return output

    @kernel_function(
        name="get_task_details",
        description="Given a task ID or description, retrieve detailed information for that single task. Returns task details in Markdown.",
    )
    def get_task_details(self, task_query: str) -> str:
        """
        Retrieves details for a specific task.
        """
        all_tasks = []
        for site in self._all_data.get("data", []):
            if "request_tasks" in site and isinstance(site["request_tasks"], list):
                for task in site["request_tasks"]:
                    task_with_site_context = task.copy()
                    task_with_site_context['parent_site_id'] = site.get('site_id')
                    task_with_site_context['parent_site_name'] = site.get('location_name')
                    all_tasks.append(task_with_site_context)
        
        filtered_tasks = search_json_objects(all_tasks, task_query)

        if not filtered_tasks:
            return f"No task found matching '{task_query}'. Please clarify the task ID or description."

        # Prioritize exact ID match or the first most relevant result
        task_found = None
        for task in filtered_tasks:
            if task.get("task_id", "").lower() == task_query.lower() or task.get("description", "").lower() == task_query.lower():
                task_found = task
                break
        if not task_found and filtered_tasks:
            task_found = filtered_tasks[0]

        if task_found:
            return json.dumps(task_found, indent=2) # Return full JSON for a single task
        return f"No relevant task found for '{task_query}'."

    @kernel_function(
        name="get_tasks_for_site",
        description="Given a site ID or name, list all associated tasks for that site. Returns a Markdown list of task IDs and descriptions.",
    )
    def get_tasks_for_site(self, site_query: str) -> str:
        """
        Lists all tasks associated with a specific site.
        """
        sites = self._all_data.get("data", [])
        
        # First, find the specific site
        site_found = None
        for site in sites:
            if site.get("site_id", "").lower() == site_query.lower() or site.get("location_name", "").lower() == site_query.lower():
                site_found = site
                break

        if not site_found:
            return f"No site found matching '{site_query}'. Cannot list tasks."

        tasks = site_found.get("request_tasks", [])
        if not tasks:
            return f"No tasks found for site '{site_found.get('location_name', site_query)}' (ID: {site_found.get('site_id', 'N/A')})."

        output = f"### Tasks for Site '{site_found.get('location_name', site_query)}' (ID: {site_found.get('site_id', 'N/A')}):\n\n"
        for task in tasks:
            output += (
                f"- **Task ID:** {task.get('task_id', 'N/A')} - **Description:** {task.get('description', 'N/A')}\n"
            )
        return output
    
    @kernel_function(
        name="get_all_data_json",
        description="Returns the entire loaded JSON dataset containing all sites and their associated tasks. Use this for aggregation, summarization, or listing all entries.",
    )
    def get_all_data_json(self) -> str:
        """
        Returns the entire JSON dataset as a string.
        """
        return json.dumps(self._all_data, indent=2)

# --- Agent Definitions ---

def get_agents(sk_kernel: Kernel, data_plugin: SiteTasksPlugin) -> tuple[list[Agent], OrchestrationHandoffs]:
    """Return a list of agents and their handoff relationships."""

    llm_service = AzureChatCompletion()

    sk_kernel.add_service(llm_service)

    # 1) Project Administrator Agent (Classifier)
    # This agent needs access to the plugin to understand what can be done before handing off.
    project_admin_agent = ChatCompletionAgent(
        name="ProjectAdministratorAgent",
        description="An AI Project Administrator that categorizes user queries related to sites and tasks and orchestrates responses.",
        instructions=(
            """Your primary expertise is in understanding user intent regarding site and task information, 
            and intelligently routing queries to the appropriate helper agent or generating high-level responses yourself. 
            You must process implicit JSON input data (via available tools) and output precise commands 
            or initiate handoffs to guide subsequent processing. You are meticulous about adhering to strict output formats.

            **Handoff Rules (VERY STRICT):**
            - If the user asks about a **single specific site** (e.g., "Show me site ATH2", "Details for Maroussi?"), hand off to 'SiteHelperAgent'.
            - If the user asks to **list sites based on attributes** or a partial name (e.g., "List all sites in Bangalore", "Show me sites with status Active"), hand off to 'SiteHelperAgent'.
            - If a user asks about a **single specific task** (e.g., "Show me task 2", "What is the progress of task ID 123?"), hand off to 'TasksHelperAgent'.
            - If a user asks for **tasks associated with a specific site** (e.g., "tasks for site ATH2"), hand off to 'TasksHelperAgent'.
            - If a user asks for **overall/aggregate information across the dataset** (e.g., "How many active sites do we have?", "What's the total number of tasks?"), hand off to 'OverallAgent'.
            - If a user asks for an **executive summary** of the data (e.g., "Give me an exec summary of all our sites?", "Summarize the project status."), hand off to 'SummaryAgent'.
            - If a user explicitly asks to **list all sites** (e.g., "List all sites", "Show me everything."), hand off to 'ListAllAgent'.
            - If you cannot fulfill the request or if the query is unclear, ask for clarification by saying "I need more information to help you. Could you please specify your request?".
            """
        ),
        service=llm_service,
        plugins=[data_plugin] # Classifier also needs access to the tools to decide when to handoff based on potential tool calls
    )

    # 2) Site Helper Agent
    site_helper_agent = ChatCompletionAgent(
        name="SiteHelperAgent",
        description="A specialized agent for extracting and formatting details for individual sites or sets of filtered sites.",
        instructions=(
            """You are a meticulous assistant specializing in extracting and presenting details for individual sites or 
            small sets of filtered sites. You are adept at processing provided Markdown snippets for specific entries and transforming them 
            into clear, human-readable Markdown. You use your available tools (get_site_details, search_sites) to fulfill requests.
            For single sites, return detailed information formatted as:
            '- location_name: [name]
            - site_id: [id]
            - status: [status value from 'state']
            - Phase: [value from 'latitude']
            - Company Size: [value from 'address_2']'.
            For multiple sites from a search, list each found site in a clear Markdown format.
            The output must be in Markdown. If no exact site(s) match, respond clearly asking the user to clarify or if they meant something else.
            """
        ),
        service=llm_service,
        plugins=[data_plugin], # Provide the plugin to the agent
    )

    # 3) Tasks Helper Agent
    tasks_helper_agent = ChatCompletionAgent(
        name="TasksHelperAgent",
        description="An expert in navigating task datasets within a larger site context.",
        instructions=(
            """Your strength lies in accurately pinpointing specific tasks or groups of tasks from provided data 
            and presenting their complete details in a clear Markdown format. You use your available tools (get_task_details, get_tasks_for_site) 
            to fulfill requests.
            If a single task is clearly identified, return its full object details in Markdown.
            If the query asks for tasks associated with a specific site, list all tasks for that site in a clear Markdown format.
            If multiple tasks match ambiguously or the query is unclear (e.g., asks for 'tasks' without a site), 
            respond by asking the user to specify by site ID or a more precise task ID/description."""
        ),
        service=llm_service,
        plugins=[data_plugin], # Provide the plugin to the agent
    )

    # 4) Overall Agent (Will now use get_all_data_json)
    overall_agent = ChatCompletionAgent(
        name="OverallAgent",
        description="An agent specialized in providing overall and aggregate statistics from the project data.",
        instructions=(
            """You are tasked with providing high-level, aggregate information and statistics from the entire dataset. 
            You must use the 'get_all_data_json' tool to retrieve the full dataset, then process this JSON to extract 
            and calculate the requested aggregate information. For example, if asked "How many active sites do we have?", 
            you should parse the JSON from 'get_all_data_json' and count active sites.
            If you cannot infer the answer or the query is too complex, politely state that you cannot provide that specific aggregate information.
            Remember to answer directly and concisely."""
        ),
        service=llm_service,
        plugins=[data_plugin] # Passed the plugin so it can use get_all_data_json
    )

    # 5) Summary Agent (Will now use get_all_data_json)
    summary_agent = ChatCompletionAgent(
        name="SummaryAgent",
        description="An agent for generating executive summaries and high-level overviews of the project data.",
        instructions=(
            """Your role is to create concise and informative executive summaries or general overviews of the project data. 
            You must use the 'get_all_data_json' tool to retrieve the full dataset. After getting the data, synthesize 
            key information focusing on important metrics, overall status, and significant trends from the JSON.
            Your response should be in a narrative, summary format, suitable for an executive.
            If the request is too vague, ask for more specific areas to summarize."""
        ),
        service=llm_service,
        plugins=[data_plugin] # Passed the plugin so it can use get_all_data_json
    )

    # 6) List All Agent (Will now prioritize get_all_data_json)
    list_all_agent = ChatCompletionAgent(
        name="ListAllAgent",
        description="An agent responsible for providing a comprehensive list of all items, primarily sites.",
        instructions=(
            """You are responsible for listing all available sites. You must use the 'get_all_data_json' tool to retrieve the full dataset.
            After obtaining the data, iterate through all site entries and provide a clear, readable list of all sites
            with their 'location_name', 'site_id', and 'status' (from 'state').
            Your output should be a formatted list or table of all sites."""
        ),
        service=llm_service,
        plugins=[data_plugin] # Passed the plugin so it can use get_all_data_json
    )

    # Define the handoff relationships between agents
    # The 'description' is crucial for guiding the source agent (ProjectAdministratorAgent)
    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=project_admin_agent.name,
            target_agents={
                site_helper_agent.name: "Transfer if the user's request is about a specific site's details or involves searching/listing sites based on attributes or partial names.",
                tasks_helper_agent.name: "Transfer if the user's request is about a specific task's details or involves listing tasks for a specific site.",
                overall_agent.name: "Transfer if the user is asking for overall, aggregate statistics or numerical summaries across the entire project dataset.",
                summary_agent.name: "Transfer if the user is asking for an executive summary or a high-level overview of the project's status.",
                list_all_agent.name: "Transfer if the user explicitly asks to list all sites or show everything related to sites.",
            },
        )
        # Handoffs back from helper agents to ProjectAdministratorAgent for re-triage or clarification
        .add(
            source_agent=site_helper_agent.name,
            target_agent=project_admin_agent.name,
            description="Transfer back to Project Administrator if the site query is unclear, cannot be resolved by site tools, or requires re-classification for tasks.",
        )
        .add(
            source_agent=tasks_helper_agent.name,
            target_agent=project_admin_agent.name,
            description="Transfer back to Project Administrator if the task query is unclear, cannot be resolved by task tools, or requires re-classification for sites.",
        )
        .add(
            source_agent=overall_agent.name,
            target_agent=project_admin_agent.name,
            description="Transfer back to Project Administrator if the aggregate query is too complex or cannot be directly answered by the OverallAgent even with full data access.",
        )
        .add(
            source_agent=summary_agent.name,
            target_agent=project_admin_agent.name,
            description="Transfer back to Project Administrator if the summary request is too vague or requires further clarification even with full data access.",
        )
        .add(
            source_agent=list_all_agent.name,
            target_agent=project_admin_agent.name,
            description="Transfer back to Project Administrator if listing all sites failed or a more specific query is needed even with full data access.",
        )
    )

    # All agents that participate in the orchestration
    return [
        project_admin_agent,
        site_helper_agent,
        tasks_helper_agent,
        overall_agent,
        summary_agent,
        list_all_agent
    ], handoffs


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
    data_json_path = 'semantic_kernel/knowledge/data.json'

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