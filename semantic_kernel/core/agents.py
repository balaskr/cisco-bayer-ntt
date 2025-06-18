import logging
import os

from core.plugins import SiteTasksPlugin
from semantic_kernel.agents import (Agent, ChatCompletionAgent,
                                    OrchestrationHandoffs)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import (AuthorRole, ChatMessageContent,
                                      FunctionCallContent,
                                      FunctionResultContent)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.logging import setup_logging

from dotenv import load_dotenv;load_dotenv()


setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)


# --- Agent Definitions ---


def get_agents(sk_kernel: Kernel, data_plugin: SiteTasksPlugin) -> tuple[list[Agent], OrchestrationHandoffs]:
    llm_service = AzureChatCompletion()
    sk_kernel.add_service(llm_service)

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
        plugins=[data_plugin]
    )

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
        plugins=[data_plugin],
    )

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
        plugins=[data_plugin],
    )

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
        plugins=[data_plugin]
    )

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
        plugins=[data_plugin]
    )

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
        plugins=[data_plugin]
    )

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

    return [
        project_admin_agent,
        site_helper_agent,
        tasks_helper_agent,
        overall_agent,
        summary_agent,
        list_all_agent
    ], handoffs


def agent_response_callback(message: ChatMessageContent) -> None:
    logging.info(f"\n--- Agent Response ({message.name}) ---")
    if message.content:
        logging.info(f"Content: {message.content}")
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            logging.info(f"Function Call: '{item.name}' with arguments '{item.arguments}'")
        if isinstance(item, FunctionResultContent):
            result_str = str(item.result)
            if len(result_str) > 500:
                logging.info(f"Function Result from '{item.name}':\n{result_str[:500]}...\n(Truncated for display)")
            else:
                logging.info(f"Function Result from '{item.name}':\n{result_str}")
    logging.info("-------------------------------------")


# This function is not used in the HTTP trigger, but kept for completeness of the original SK code.
async def human_response_function() -> ChatMessageContent:
    user_input = "Agent requires input. This won't be used in HTTP trigger."
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)
# --- Agent Definitions ---

def get_agents(sk_kernel: Kernel, data_plugin: SiteTasksPlugin) -> tuple[list[Agent], OrchestrationHandoffs]:
    llm_service = AzureChatCompletion(
        service_id="default",
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    sk_kernel.add_service(llm_service)

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
        plugins=[data_plugin]
    )

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
        plugins=[data_plugin],
    )

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
        plugins=[data_plugin],
    )

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
        plugins=[data_plugin]
    )

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
        plugins=[data_plugin]
    )

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
        plugins=[data_plugin]
    )

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

    return [
        project_admin_agent,
        site_helper_agent,
        tasks_helper_agent,
        overall_agent,
        summary_agent,
        list_all_agent
    ], handoffs


def agent_response_callback(message: ChatMessageContent) -> None:
    logging.info(f"\n--- Agent Response ({message.name}) ---")
    if message.content:
        logging.info(f"Content: {message.content}")
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            logging.info(f"Function Call: '{item.name}' with arguments '{item.arguments}'")
        if isinstance(item, FunctionResultContent):
            result_str = str(item.result)
            if len(result_str) > 500:
                logging.info(f"Function Result from '{item.name}':\n{result_str[:500]}...\n(Truncated for display)")
            else:
                logging.info(f"Function Result from '{item.name}':\n{result_str}")
    logging.info("-------------------------------------")


# This function is not used in the HTTP trigger, but kept for completeness of the original SK code.
async def human_response_function() -> ChatMessageContent:
    user_input = "Agent requires input. This won't be used in HTTP trigger."
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)
