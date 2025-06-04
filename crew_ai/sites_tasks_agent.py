from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
import json

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=""
)

embedder= {
        "provider": "google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": ""
        }
    }

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
import json
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource

def run_multimodal_query(user_query: str, hidden_json: dict, chat_history: list) -> str:
    print("\n--- START MULTIMODAL QUERY ---")
    print(f"User query: {user_query}")
    print("Setting up Step 1: Classification")

    # Step 1 - Classification
    json_ks_step1 = JSONKnowledgeSource(file_paths=["data.json"])

    classify_task = Task(
        description=user_query,
        agent=classifier,
        expected_output='A label for the query to classify it.'
    )

    crew_step1 = Crew(
        agents=[classifier],
        tasks=[classify_task],
        knowledge_sources=[json_ks_step1],
        embedder=embedder,
        verbose=True
    )

    classify_result = crew_step1.kickoff()
    print(f"Classification result: {classify_result.raw}")

    label = classify_result.raw
    context_file = "data.json"
    agent = None
    description = user_query

    print("Analyzing classification label...")

    if label.startswith("SITE:"):
        query = label[len("SITE:"):].strip()
        print(f"[SITE mode] Query: {query}")
        filtered = [s for s in hidden_json["data"] if query.lower() in json.dumps(s).lower()]
        with open("knowledge/filtered_site.json", "w") as f:
            json.dump(filtered, f)
        context_file = "filtered_site.json"
        agent = site_helper
        description = query

    elif label.startswith("TASK:"):
        query = label[len("TASK:"):].strip()
        print(f"[TASK mode] Query: {query}")
        filtered = []
        for s in hidden_json["data"]:
            filtered.extend([t for t in s.get("request_tasks", []) if query.lower() in json.dumps(t).lower()])
        with open("knowledge/filtered_task.json", "w") as f:
            json.dump(filtered, f)
        context_file = "filtered_task.json"
        agent = tasks_helper
        description = query

    elif label.startswith("OVERALL:"):
        query = label[len("OVERALL:"):].strip()
        print(f"[OVERALL mode] Query: {query}")
        agent = overall_agent
        description = query

    elif label == "SUMMARY":
        print("[SUMMARY mode]")
        agent = summary_agent

    else:
        print("[FALLBACK mode]")
        return label

    print(f"Preparing Step 2 with context file: {context_file} and agent: {agent.role}")

    json_ks_step2 = JSONKnowledgeSource(file_paths=[context_file])

    task2 = Task(
        description=description,
        agent=agent,
        expected_output="Answer to the query."
    )

    crew_step2 = Crew(
        agents=[agent],
        tasks=[task2],
        knowledge_sources=[json_ks_step1],
        embedder=embedder,
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
user_query = "show me details of ATH2"

# Empty chat history for this example
chat_history = []

# Call the function
response = run_multimodal_query(user_query, hidden_json, chat_history)

# Print the result
print("\nFinal Response:\n", response)
