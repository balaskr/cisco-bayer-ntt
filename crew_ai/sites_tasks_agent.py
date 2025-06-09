import json
import os
from dotenv import load_dotenv;load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM
from utils import search_json_objects

# Ensure environment variables are loaded
load_dotenv()

# Configure your LLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agent Definitions ---

# 1) Classifier Agent (UPDATED goal)
classifier = Agent(
    role="SiteTasks AI Project Administrator",
    goal=(
        """Accurately categorize user queries related to sites and tasks based on provided JSON data.
        Your output MUST be one of these exact formats:
        RELOAD, SITE:<exact user query>, TASK:<exact user query>, OVERALL:<exact user query>,
        SUMMARY, SEARCH:<exact user query>, or LISTALL.
        Strictly adhere to these output patterns without additional text, explanations, or quotes."""
    ),
    backstory=(
        """You are a highly specialized AI Project Administrator agent.
        Your primary expertise is in understanding user intent regarding site and task information,
        and intelligently routing queries to the appropriate helper or generating specific commands.
        You process JSON input data (from a 'system' prompt/context) and output precise commands
        to guide subsequent processing. You are meticulous about adhering to strict output formats."""
    ),
    llm=llm,
    verbose=False
)

# 2) Site Helper Agent (Slightly adjusted goal for clarity, handles specific/filtered sites)
site_helper = Agent(
    role="Specific Site Details Extractor and Formatter",
    goal=(
        """Given a user's query about a specific site (e.g., 'ATH2', 'site 123') or a filtered search (e.g., 'sites in Bangalore')
        and the **already filtered** JSON context of relevant site objects, identify and return the requested site(s).
        For single sites, return detailed information formatted as:
        '- location_name: [name]
        - site_id: [id]
        - status: [status value from 'state']
        - Phase: [value from 'latitude']
        - Company Size: [value from 'address_2']'.
        For multiple sites from a search, list each found site in a clear Markdown format.
        The output must be in Markdown. If no exact site(s) match within the provided filtered data,
        respond clearly asking the user to clarify the ID or name, stating that no relevant site was found."""
    ),
    backstory=(
        """You are a meticulous assistant specializing in extracting and presenting details for individual sites or small sets of filtered sites.
        You are adept at processing provided JSON snippets for specific entries and transforming them
        into clear, human-readable Markdown. You are precise with field renames (latitude to Phase, address_2 to Company Size)
        and extract 'state' for 'status'."""
    ),
    llm=llm,
    verbose=False
)

# 3) Tasks Helper Agent
tasks_helper = Agent(
    role="Task Details Extractor and Formatter",
    goal=(
        """Given a user's specific task query (e.g., 'task 456', 'tasks for site ATH2') and the **already filtered** JSON context
        of relevant tasks/sites, identify the requested task(s).
        If a single task is clearly identified, return its full object details in Markdown.
        If the query asks for tasks associated with a specific site, list all tasks for that site in a clear Markdown format.
        If multiple tasks match ambiguously or the query is unclear (e.g., asks for 'tasks' without a site),
        respond by asking the user to specify by site ID or a more precise task ID/description."""
    ),
    backstory=(
        """You are an expert in navigating task datasets within a larger site context.
        Your strength lies in accurately pinpointing specific tasks or groups of tasks from provided data
        and presenting their complete details in a clear Markdown format. You can handle requests for individual tasks or all tasks related to a specific site."""
    ),
    llm=llm,
    verbose=False
)

# 4) Overall Agent 
overall_agent = Agent(
    role="Aggregate Data Analyst",
    goal=(
        """Given the entire JSON of all sites and tasks, answer aggregate or high-level questions
        about counts of sites, tasks, their statuses, or any other overall parameter requested.
        Present your findings concisely in Markdown format.
        If the user's request is not clear, cannot be answered from the provided data,
        or is too broad for aggregation, clearly ask for clarification."""
    ),
    backstory=(
        """You possess a holistic view of the entire dataset, excelling at synthesizing information
        to provide high-level summaries and statistical breakdowns. You are the go-to for
        bird's-eye insights across all sites and tasks, capable of identifying trends and totals."""
    ),
    llm=llm,
    verbose=False
)

# 5) Summary Agent
summary_agent = Agent(
    role="Executive Summary Generator",
    goal=(
        """Given the entire JSON of all sites and tasks, generate a concise executive summary
        (aim for 3-4 paragraphs). The summary MUST include:
        a brief description of the overall data/project,
        the total number of sites, the total number of tasks, and a breakdown of their statuses.
        Format the summary in clear, readable Markdown, suitable for an executive audience."""
    ),
    backstory=(
        """You are a skilled summarizer, capable of distilling large amounts of complex data
        into actionable and digestible narratives. Your summaries are executive-level,
        providing key insights at a glance, highlighting important metrics and status overviews."""
    ),
    llm=llm,
    verbose=False
)

# 6) NEW: List All Agent
list_all_agent = Agent(
    role="Comprehensive Site Lister",
    goal=(
        """Given the complete JSON dataset of all sites, generate a comprehensive list of all sites.
        Present each site's 'location_name', 'site_id', and 'status' (from 'state')
        in a clear Markdown table format. Ensure all available sites are included in the list."""
    ),
    backstory=(
        """You are an expert in cataloging and presenting exhaustive lists of data entries.
        Your primary function is to meticulously extract details for every site available in the provided dataset
        and organize them into a clean, easy-to-read Markdown table, prioritizing clarity and completeness."""
    ),
    llm=llm,
    verbose=False
)


# --- Main function to run the multimodal query ---

def run_sites_tasks_agent_query(user_query: str, context_json: dict, chat_history: list = None) -> str:
    if chat_history is None:
        chat_history = []

    print("Setting up Step 1: Classification")

    # --- UPDATED CLASSIFICATION INSTRUCTIONS ---
    classification_instructions = f"""
    Analyze the user's query and the provided JSON data context.
    Your goal is to categorize the user's request and output a specific, exact command.

    **JSON Data Context (Implicitly available to you as an LLM for reasoning):**
    - The 'data' variable in the JSON object lists all the site objects.
    - The 'request_tasks' variable within site objects lists all the tasks or projects associated with each containing site.

    **Instructions for Classification and Output Format (VERY STRICT):**

    1.  **Understand the User's Request:** Your primary responsibility is to confirm that the user's query relates to retrieving sites, site tasks, or general task information for the current logged-in client.

    2.  **Missing Data Check (RELOAD):**
        - If a user asks anything about sites, site tasks, tasks, or projects, AND the necessary data for processing that query is *not* implicitly available in the context (e.g., it's empty, malformed, or insufficient for the query), respond ONLY with `RELOAD`.
        - Example: User asks "Show me site 1" but `context_json` is empty or clearly lacking site data.
        - DO NOT provide additional information or quotation marks around `RELOAD`.

    3.  **Single Site Details (SITE:<query>):**
        - If the user asks about a **single specific site** by its known ID or a very precise name, and they are looking for its direct details (e.g., "Show me site 2", "What is the status of ATH2?", "Details for Maroussi?"), respond ONLY with `SITE:<with the user request exactly as given>`.
        - This implies a direct lookup of a known entity.
        - DO NOT provide additional information or quotation marks.

    4.  **Overall/Aggregate Statistics (OVERALL:<query>):**
        - If a user asks for **aggregate information, counts, or summaries across the entire dataset** (e.g., "How many active sites do we have?", "What's the total number of tasks?", "Give me a breakdown of site statuses?"), respond ONLY with `OVERALL:<with the user request exactly as given>`.
        - This is for numerical summaries or high-level overviews, not itemized lists.
        - DO NOT provide additional information or quotation marks.

    5.  **Single Task Details (TASK:<query>):**
        - If a user asks about a **single specific task** by its ID or a very precise description (e.g., "Show me task 2", "What is the progress of task ID 123?"), respond ONLY with `TASK:<with the user request exactly as given>`.
        - If a user asks for tasks in general (e.g., "List tasks", "What tasks do we have?") but does NOT specify a site, your response should still be `TASK:`, but you should include a clear prompt for the user to specify a site within the query, e.g., `TASK:Please specify which site you want tasks for.`.
        - DO NOT provide additional information or quotation marks.

    6.  **Site Listing / Filtered Search by Attributes (SEARCH:<query>):**
        - If the user asks to **list sites based on a specific attribute or location**, or requests a search that would result in multiple site details, or is looking for a site based on a partial/fuzzy name/location (e.g., "List all sites in Bangalore", "Show me all sites with status Active", "Show me sites located in Europe"), respond ONLY with `SEARCH:<with the user request exactly as given>`.
        - This category is for filtering and listing specific site objects from the dataset based on *criteria*.
        - DO NOT provide additional information or quotation marks.

    7.  **List All Sites (LISTALL):**
        - If the user explicitly asks to **list ALL sites** without any filtering criteria (e.g., "list all my sites", "show me all sites", "list all entries", "all sites in a table"), respond ONLY with `LISTALL`.
        - DO NOT provide additional information or quotation marks.

    8.  **Executive Summary (SUMMARY):**
        - If a user asks for an executive summary or a general summary of the entire operation (e.g., "Give me an exec summary of all our sites", "Summarize everything"), respond ONLY with `SUMMARY`.
        - DO NOT provide additional information or quotation marks.

    ---
    **Current User Query to Classify:** "{user_query}"
    **Implicit JSON Context (for your reference, not for direct output):**
    {json.dumps(context_json, indent=2)}
    """

    classify_task = Task(
        description=classification_instructions,
        agent=classifier,
        expected_output=(
            """A single, uppercase classification label and query following the strict formats:
            'RELOAD', 'SITE:<query>', 'TASK:<query>', 'OVERALL:<query>', 'SUMMARY', 'SEARCH:<query>', or 'LISTALL'.
            No extra text, no markdown, and no quotes. Only the exact command."""
        )
    )

    crew_step1 = Crew(
        agents=[classifier],
        tasks=[classify_task],
        verbose=True,
        process=Process.sequential
    )

    label = crew_step1.kickoff().raw # Use .raw to get the string output from kickoff
    print(f"Classification result: {label}")

    agent_to_delegate_to = None
    task_description_for_helper = user_query
    filtered_data = None

    print("Analyzing classification label...")

    if label.strip() == "RELOAD":
        print("[RELOAD mode] - Requesting data reload.")
        return "RELOAD"

    elif label.startswith("SITE:") or label.startswith("SEARCH:"):
        query = label[label.find(":")+1:].strip()
        print(f"[{'SITE' if label.startswith('SITE:') else 'SEARCH'} mode] Query: {query}")

        if context_json and "data" in context_json and isinstance(context_json["data"], list):
            filtered_data = search_json_objects(context_json["data"], query)
            if not filtered_data:
                print(f"No relevant site data found for query: '{query}'. Passing empty list to helper.")
                filtered_data = [] # Explicitly set to empty list for helper processing
        else:
            print("Warning: 'data' key not found or not a list in context_json. Skipping filtering.")
            filtered_data = []

        agent_to_delegate_to = site_helper
        task_description_for_helper = (
            f"""The user queried a site or asked to search/list sites: '{query}'.
            You have been provided with **pre-filtered JSON data** relevant to this query.
            Process this provided data to find the requested site(s) and return details as specified in your goal.
            If no relevant site(s) are found in the filtered data, report it as not found or ask for clarification.

            **Provided Filtered JSON Context:**
            {json.dumps(filtered_data, indent=2)}"""
        )

    elif label.startswith("TASK:"):
        query = label[len("TASK:"):].strip()
        print(f"[TASK mode] Query: {query}")

        if context_json and "data" in context_json and isinstance(context_json["data"], list):
            all_tasks = []
            for site in context_json["data"]:
                if "request_tasks" in site and isinstance(site["request_tasks"], list):
                    for task in site["request_tasks"]:
                        task_with_site_context = task.copy()
                        task_with_site_context['parent_site_id'] = site.get('site_id')
                        task_with_site_context['parent_site_name'] = site.get('location_name')
                        all_tasks.append(task_with_site_context)
            
            filtered_data = search_json_objects(all_tasks, query)
            if not filtered_data:
                print(f"No relevant task data found for query: '{query}'. Passing empty list to helper.")
                filtered_data = []
        else:
            print("Warning: 'data' key not found or not a list in context_json. Skipping filtering.")
            filtered_data = []

        agent_to_delegate_to = tasks_helper
        task_description_for_helper = (
            f"""The user queried a task: '{query}'.
            You have been provided with **pre-filtered JSON data** relevant to this query (may contain tasks from various sites).
            Find the requested task(s) within this provided data and return their details as specified in your goal.
            If the task is not found, report it as not found or ask for clarification.

            **Provided Filtered JSON Context:**
            {json.dumps(filtered_data, indent=2)}"""
        )

    elif label.startswith("OVERALL:"):
        query = label[len("OVERALL:"):].strip()
        print(f"[OVERALL mode] Query: {query}")
        agent_to_delegate_to = overall_agent
        task_description_for_helper = (
            f"""The user asked an overall/aggregate question: '{query}'.
            Analyze the entire JSON context to answer this question as specified in your goal.

            **JSON Context:**
            {json.dumps(context_json, indent=2)}"""
        )

    elif label == "SUMMARY":
        print("[SUMMARY mode] - Generating executive summary.")
        agent_to_delegate_to = summary_agent
        task_description_for_helper = (
            f"""The user requested an executive summary.
            Generate a 3-4 paragraph summary based on the entire JSON context as specified in your goal.

            **JSON Context:**
            {json.dumps(context_json, indent=2)}"""
        )
    
    elif label == "LISTALL": # NEW LISTALL ROUTING
        print("[LISTALL mode] - Listing all sites.")
        agent_to_delegate_to = list_all_agent
        # For LISTALL, pass the ENTIRE site data, no filtering.
        task_description_for_helper = (
            f"""The user requested a list of ALL sites.
            You have been provided with the complete JSON context for all sites.
            Generate a comprehensive list of all sites as specified in your goal.

            **Provided Full JSON Context:**
            {json.dumps(context_json.get("data", []), indent=2)}""" 
        )

    else:
        print(f"[FALLBACK mode] Unhandled classification label: '{label}'")
        return label

    print(f"Preparing Step 2 with agent: {agent_to_delegate_to.role}")

    helper_task = Task(
        description=task_description_for_helper,
        agent=agent_to_delegate_to,
        expected_output="""An answer to the query, or a specific clarification request. NO JSON OR CODEBLOCKS WHATSOEVER"""
    )

    crew_step2 = Crew(
        agents=[agent_to_delegate_to],
        tasks=[helper_task],
        verbose=True,
        process=Process.sequential
    )

    print("Kicking off Step 2 task...\n")
    final_response = crew_step2.kickoff()


    if final_response.raw.startswith('```') and final_response.raw.endswith("```"):
        if "markdown" in final_response.raw:
            return final_response.raw[11:-3]
        return final_response.raw[3:-3]

    return final_response


# Example usage (for testing this file directly)
if __name__ == "__main__":
    # Load JSON (site + task data)
    try:
        with open("knowledge/data.json", "r") as f:
            context_json = json.load(f)
    except FileNotFoundError:
        print("Error: knowledge/data.json not found. Please ensure the file exists.")
        context_json = {"data": []}

    # Example user queries
    user_query_all_sites_table = "list all my sites in a table" 
    user_query1 = "show me the tasks of ATH2"
    user_query2 = "give me an exec summary of all our sites"
    user_query3 = "Show me site ATH2"
    user_query4 = "How many sites are currently active?"
    user_query5 = "What is task 99?"
    user_query6 = "List all sites in bangalore"
    user_query7 = "Show me Maroussi"
    user_query8 = "Show me all sites with status Active"

    chat_history = []

    print(f"\n--- Running Query: '{user_query_all_sites_table}' ---")
    response = run_sites_tasks_agent_query(user_query_all_sites_table, context_json, chat_history)
    print(f"\nFinal Response:\n{response}")