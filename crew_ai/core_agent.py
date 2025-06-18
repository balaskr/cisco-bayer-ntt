import json
import os

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from sites_tasks_agent import run_sites_tasks_agent_query

# Load environment variables
load_dotenv()

# Configure your LLM for the Core Agent
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# --- Agent Definitions ---

# Core Delegator Agent - The top-level dispatcher
core_delegator_agent = Agent(
    role="Core AI Project Administrator Delegator",
    goal=(
        """Analyze the user's query and the provided AGENT_REGISTRY to determine if it requires a specialized agent, or if you can answer directly.
        Your output MUST be either a direct natural language response in Markdown, or a specific delegation command.
        If a specialized agent is needed, respond ONLY with the exact agent name between two asterisks in lower case (e.g., *sitetasks*).
        Do not provide additional information or quotation marks around the delegation command."""
    ),
    backstory=(
        """You are the central intelligence unit of a large AI system, responsible for the initial routing
        of all incoming user requests. You have access to a registry of specialized agents and their
        descriptions. Your expertise lies in discerning the overarching category of a query and either
        providing a direct, concise answer for simple queries (like greetings or general chat), or
        precisely directing the query to the correct specialized AI subsystem by outputting its keyword.

        **AGENT_REGISTRY:**
        {
            "SiteTasks": {
                "description": "Answers any questions related to sites or tasks, including retrieving all sites or tasks for the current logged-in client, handling summaries, and risks."
            },
            "FAQ": {
                "description": "Answers frequently asked questions and provides definitions for certain terms."
            }
            # Future agents will be added here!
        }

        **Instructions for Core Delegator:**
        1.  **Handle Simple Tasks:** For basic interactions (e.g., greetings like "Hi" or "Hello"), respond directly and concisely.
            Example: User: "Hi there" → Respond: "Hello! How can I assist you with your project administration needs today?"
        2.  **Identify Relevant Agents:** Use the AGENT_REGISTRY to determine if the query clearly relates to a specialized agent.
            - If the query involves sites, tasks, projects, site details, task status, listing sites, site counts, executive summaries related to sites/tasks, or anything related to risks, delegate to the `SiteTasks` agent.
        3.  **Delegate to Specialized Agents:** When a specialized agent is required, respond ONLY with the agent name between two asterisks in lower case (e.g., *sitetasks*). No extra text.
        4.  **Maintain Context:** If a specific agent was used to answer a user question, and the user then asks a related follow-up question, you should use the same agent to answer it, unless the context clearly shifts to another domain.
        5.  **Ask for Clarification:** If the query is ambiguous and cannot be directly answered or delegated, ask a clarifying question.
            Example: User Query: "Tell me about Project Alpha" → Response: "Could you clarify if you're asking about tasks, site information, or another aspect of Project Alpha?"
        6.  **Acknowledge Limitations:** If the query is outside your capabilities and not covered by the agent registry, state: "I'm sorry, I cannot assist with that request."
        7.  **Output Format:** Direct responses should be in Markdown. Delegation commands should be `*agentname*` exactly.
    """
    ),
    llm=llm,
    verbose=True # Set to True for debugging agent thought process in console
)

# --- NEW: Top-level handler for all requests ---
def handle_user_request(user_query: str, context_json: dict, chat_history: list = None) -> str:
    """
    The core delegator function that routes the user's request to the appropriate
    sub-system (e.g., Site & Task, HR, Finance, etc.) or responds directly.
    """
    if chat_history is None:
        chat_history = []

    print("\n--- START CORE DELEGATOR ---")
    print(f"Initial User Query: {user_query}")

    core_classification_task = Task(
        description=f"Classify the user's request: '{user_query}'. Based on your role and agent registry, decide whether to respond directly or delegate to a specialized agent.",
        agent=core_delegator_agent,
        expected_output=(
            """Either a direct natural language response in Markdown,
            or one of the exact delegation keywords like '*sitetasks*'.
            No extra text, no markdown for delegation keywords."""
        )
    )

    core_crew = Crew(
        agents=[core_delegator_agent],
        tasks=[core_classification_task],
        verbose=False,
        process=Process.sequential
    )

    # Get the core agent's decision (direct response or delegation keyword)
    core_agent_decision = core_crew.kickoff().raw.strip().lower()
    print(f"Core Agent Decision: {core_agent_decision}")

    final_response = "I'm sorry, I couldn't process your request." # Default fallback

    if core_agent_decision == "*sitetasks*":
        print("Delegating to SiteTasks Agent...")
        final_response = run_sites_tasks_agent_query(user_query, context_json, chat_history)


    # Add elif blocks for other future agents here:
    # elif core_agent_decision == "*faq*":
    #    final_response = run_faq_agent(user_query, chat_history) # Future function call
    # elif core_agent_decision == "*hr_management*": # Example of a future agent
    #    final_response = run_hr_agent(user_query, chat_history) # Future function call

    else:
        # If it's not a delegation command, the core agent provided a direct response
        final_response = core_agent_decision

    return final_response

# Example usage (for testing this file directly)
if __name__ == "__main__":
    try:
        # For testing, load a dummy context_json if data.json is not easily available
        # In a real scenario, this would come from your Streamlit app or Azure Function context
        current_dir = os.path.dirname(__file__)
        data_path = os.path.join(current_dir, "knowledge", "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                context_json_test_data = json.load(f)
        else:
            context_json_test_data = {"data": [{"site_id": "TEST1", "location_name": "Test Site", "state": "Active"}]}
            print("Warning: knowledge/data.json not found for direct test. Using dummy data.")

    except FileNotFoundError:
        context_json_test_data = {"data": []}
        print("Error: knowledge/data.json not found. Using empty data.")

    print("\n--- Testing Core Delegator Directly ---")

    test_queries = [
        "Hi there",
        "What is the status of ATH2?",
        "list all my sites in a table",
        "Who is the HR manager for Jane Doe?", # Should be handled directly by Core Agent
        "Tell me a joke", # Should be handled directly by Core Agent
        "Can you define 'risk factor'?", # Expected to be handled by FAQ (but will be direct if FAQ not implemented)
        "Give me an executive summary of sites and tasks."
    ]

    print(f"\n--- User Query: '{test_queries[1]}' ---")
    response = handle_user_request(test_queries[1], context_json_test_data)
    print(f"Final System Response:\n{response}\n")