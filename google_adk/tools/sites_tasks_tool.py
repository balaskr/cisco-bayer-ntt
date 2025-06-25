import json
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import FunctionTool
from google.adk.tools.base_toolset import BaseTool, BaseToolset

from ..utils import search_json_objects

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def call_api() -> Dict[str, Any]:
    """Loads data from the knowledge/data.json file."""
    try:
        with open("google_adk/knowledge/data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("knowledge/data.json not found.")
        return {}

# Tool Functions
def get_site_details(site_query: str) -> str:
    all_data = call_api()
    logger.info(f"Getting site details for query: '{site_query}'")
    sites = all_data.get("data", [])
    filtered_sites = search_json_objects(sites, site_query)

    if not filtered_sites:
        logger.warning(f"No site found matching '{site_query}'.")
        return f"No site found matching '{site_query}'. Please clarify the site ID or name."

    site_found = None
    for site in filtered_sites:
        if (
            site.get("site_id", "").lower() == site_query.lower()
            or site.get("location_name", "").lower() == site_query.lower()
        ):
            site_found = site
            break
    if not site_found and filtered_sites:
        site_found = filtered_sites[0]

    if site_found:
        logger.info(f"Found site: {site_found.get('location_name', 'N/A')}")
        return (
            f"- location_name: {site_found.get('location_name', 'N/A')}\n"
            f"- site_id: {site_found.get('site_id', 'N/A')}\n"
            f"- status: {site_found.get('state', 'N/A')}\n"
            f"- Phase: {site_found.get('latitude', 'N/A')}\n"
            f"- Company Size: {site_found.get('address_2', 'N/A')}"
        )

    logger.warning(f"No relevant site found for '{site_query}'.")
    return f"No relevant site found for '{site_query}' within the provided data."


def search_sites(query: str) -> str:
    all_data = call_api()
    logger.info(f"Searching sites with query: '{query}'")
    sites = all_data.get("data", [])
    filtered_sites = search_json_objects(sites, query)

    if not filtered_sites:
        logger.warning(f"No sites found for query: '{query}'.")
        return f"No sites found matching the criteria: '{query}'."

    logger.info(f"Found {len(filtered_sites)} sites for query: '{query}'.")
    output = "### Found Sites:\n\n"
    for site in filtered_sites:
        output += (
            f"- **{site.get('location_name', 'N/A')}** (ID: {site.get('site_id', 'N/A')}) - Status: {site.get('state', 'N/A')}\n"
        )
    return output


def get_task_details(task_query: str) -> str:
    all_data = call_api()
    logger.info(f"Getting task details for query: '{task_query}'")
    all_tasks = []
    for site in all_data.get("data", []):
        if "request_tasks" in site and isinstance(site["request_tasks"], list):
            for task in site["request_tasks"]:
                task_with_site_context = task.copy()
                task_with_site_context["parent_site_id"] = site.get("site_id")
                task_with_site_context["parent_site_name"] = site.get("location_name")
                all_tasks.append(task_with_site_context)

    filtered_tasks = search_json_objects(all_tasks, task_query)

    if not filtered_tasks:
        logger.warning(f"No task found for query: '{task_query}'.")
        return f"No task found matching '{task_query}'. Please clarify the task ID or description."

    task_found = None
    for task in filtered_tasks:
        if (
            task.get("task_sys_id", "").lower() == task_query.lower()
            or task.get("classification", "").lower() == task_query.lower()
        ):
            task_found = task
            break
    if not task_found and filtered_tasks:
        task_found = filtered_tasks[0]

    if task_found:
        logger.info(f"Found task: {task_found.get('task_sys_id', 'N/A')}")
        return json.dumps(task_found, indent=2)

    logger.warning(f"No relevant task found for '{task_query}'.")
    return f"No relevant task found for '{task_query}'."


def get_tasks_for_site(site_query: str) -> str:
    all_data = call_api()
    logger.info(f"Getting tasks for site: '{site_query}'")
    sites = all_data.get("data", [])

    site_found = None
    for site in sites:
        if (
            site.get("site_id", "").lower() == site_query.lower()
            or site.get("location_name", "").lower() == site_query.lower()
        ):
            site_found = site
            break

    if not site_found:
        logger.warning(f"Site not found for query: '{site_query}'.")
        return f"No site found matching '{site_query}'. Cannot list tasks."

    tasks = site_found.get("request_tasks", [])
    if not tasks:
        logger.warning(f"No tasks found for site: {site_found.get('location_name', 'N/A')}")
        return f"No tasks found for site '{site_found.get('location_name', site_query)}' (ID: {site_found.get('site_id', 'N/A')})."

    logger.info(f"Found {len(tasks)} tasks for site: {site_found.get('location_name', 'N/A')}")
    output = f"### Tasks for Site '{site_found.get('location_name', site_query)}' (ID: {site_found.get('site_id', 'N/A')}):\n\n"
    for task in tasks:
        output += (
            f"- **Task ID:** {task.get('task_sys_id', 'N/A')} - **Classification:** {task.get('classification', 'N/A')}\n"
        )
    return output


def get_all_data_json() -> str:
    all_data = call_api()
    logger.info("Getting all data as JSON.")
    return json.dumps(all_data, indent=2)


class SiteTasksToolset(BaseToolset):
    def __init__(self):
        self._get_site_details_tool = FunctionTool(func=get_site_details)
        self._search_sites_tool = FunctionTool(func=search_sites)
        self._get_task_details_tool = FunctionTool(func=get_task_details)
        self._get_tasks_for_site_tool = FunctionTool(func=get_tasks_for_site)
        self._get_all_data_json_tool = FunctionTool(func=get_all_data_json)
        logger.info("SiteTasksToolset initialized.")

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[BaseTool]:
        return [
            self._get_site_details_tool,
            self._search_sites_tool,
            self._get_task_details_tool,
            self._get_tasks_for_site_tool,
            self._get_all_data_json_tool,
        ]

    async def close(self) -> None:
        logger.info("SiteTasksToolset closed.")