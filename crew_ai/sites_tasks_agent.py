from crewai import Agent, Task, Crew, LLM
from utils import search_json_objects
import json
import os
from dotenv import load_dotenv;load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY")
)

# 1) Classifier Agent
classifier = Agent(
    role="SitesTasksClassifier",
    goal=(
        "Look at the user's query and the hidden JSON context (as a 'system' message). "
        "Respond *only* with one of: "
        "SITE:<exact query>, TASK:<exact query>, OVERALL:<exact query>, SUMMARY, or FALLBACK:<whatever>."
    ),
    backstory="You are a specialist in recognizing requests about sites and tasks.",
    llm=llm
)

# 2) Site Helper Agent
site_helper = Agent(
    role="SiteHelperAgent",
    goal=(
        "Given a filtered list of matching site objects and the user's site query, "
        "return that site's data in Markdown (rename latitude→Phase, address_2→Company Size). "
        "If no site matches, ask them to clarify ID or name."
    ),
    backstory="You zero in on a single site's details.",
    llm=llm
)

# 3) Tasks Helper Agent
tasks_helper = Agent(
    role="TasksHelperAgent",
    goal=(
        "Given a filtered list of matching tasks (from any site) and the user's task query, "
        "return the full task object in Markdown. If ambiguous, ask to specify site ID."
    ),
    backstory="You focus on single-task details.",
    llm=llm
)

# 4) Overall Agent
overall_agent = Agent(
    role="OverallAgent",
    goal=(
        "Given the entire JSON of sites+tasks, answer aggregate questions: counts of sites, tasks, statuses, etc., "
        "in Markdown form. If the user's request doesn't match, ask for clarification."
    ),
    backstory="You provide high-level, bird's-eye answers.",
    llm=llm
)

# 5) Summary Agent
summary_agent = Agent(
    role="SummaryAgent",
    goal=(
        "Given the entire JSON of all sites and tasks, generate a short (3-4 paragraph) executive summary: "
        "a short description, number of sites, number of tasks, their status breakdown, etc."
    ),
    backstory="You distill everything into a concise narrative.",
    llm=llm
)

def run_multimodal_query(user_query: str, hidden_json: dict, chat_history: list) -> str:
    print("\n--- START MULTIMODAL QUERY ---")
    print(f"User query: {user_query}")
    print("Setting up Step 1: Classification")

    # Step 1 - Classification

    classify_task = Task(
        description=user_query,
        agent=classifier,
        expected_output='A label for the query to classify it.'
    )

    crew_step1 = Crew(
        agents=[classifier],
        tasks=[classify_task],
        verbose=True
    )

    classify_result = crew_step1.kickoff()
    print(f"Classification result: {classify_result.raw}")

    label = classify_result.raw
    agent = None
    description = user_query

    print("Analyzing classification label...")

    if label.startswith("SITE:"):
        query = label[len("SITE:"):].strip()
        print(f"[SITE mode] Query: {query}")
        filtered = search_json_objects(hidden_json["data"],query)
        agent = site_helper
        description = query +"\n\n Context \n\n" + json.dumps(filtered)

    elif label.startswith("TASK:"):
        query = label[len("TASK:"):].strip()
        print(f"[TASK mode] Query: {query}")
        filtered = search_json_objects(hidden_json["data"],query)
        agent = tasks_helper
        description = query +"\n\n Context \n\n" + json.dumps(filtered)

    elif label.startswith("OVERALL:"):
        query = label[len("OVERALL:"):].strip()
        print(f"[OVERALL mode] Query: {query}")
        agent = overall_agent
        description = query +"\n\n Context \n\n" +  json.dumps(hidden_json["data"])

    elif label == "SUMMARY":
        print("[SUMMARY mode]")
        agent = summary_agent
        description = "\n\n Context \n\n" +  json.dumps(hidden_json["data"])

    else:
        print("[FALLBACK mode]")
        return label

    print(f"Preparing Step 2 with agent: {agent.role}")


    task2 = Task(
        description=description,
        agent=agent,
        expected_output="Answer to the query."
    )

    crew_step2 = Crew(
        agents=[agent],
        tasks=[task2],
        verbose=True
    )

    print("Kicking off Step 2 task...\n")
    final = crew_step2.kickoff()
    print("\n--- END MULTIMODAL QUERY ---\n")
    return final


import json

# Load hidden JSON (site + task data)
with open("knowledge/data.json", "r") as f:
    hidden_json = json.load(f)

# Provide user query
user_query = "show me the tasks of ATH2"
user_query2 = "give me an exec summary of all our sites"

# Empty chat history for this example
chat_history = []

# Call the function
response = run_multimodal_query(user_query2, hidden_json, chat_history)

