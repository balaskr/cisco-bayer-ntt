import json
from typing import Any, Dict

from core.utils import search_json_objects
from semantic_kernel.functions import kernel_function

from dotenv import load_dotenv;load_dotenv()
# --- Plugin Definitions ---

class SiteTasksPlugin:
    """
    A plugin to interact with site and task data.
    The entire dataset is passed to its constructor.
    """
    _all_data: Dict[str, Any]

    def __init__(self, all_data_param: Dict[str, Any]):
        self._all_data = all_data_param

    @kernel_function(
        name="get_site_details",
        description="Given a site ID or name, retrieve detailed information for that single site. Returns site details in Markdown.",
    )
    def get_site_details(self, site_query: str) -> str:
        sites = self._all_data.get("data", [])
        filtered_sites = search_json_objects(sites, site_query)

        if not filtered_sites:
            return f"No site found matching '{site_query}'. Please clarify the site ID or name."
        
        site_found = None
        for site in filtered_sites:
            if site.get("site_id", "").lower() == site_query.lower() or site.get("location_name", "").lower() == site_query.lower():
                site_found = site
                break
        if not site_found and filtered_sites:
            site_found = filtered_sites[0]

        if site_found:
            return (
                f"- location_name: {site_found.get('location_name', 'N/A')}\n"
                f"- site_id: {site_found.get('site_id', 'N/A')}\n"
                f"- status: {site_found.get('state', 'N/A')}\n"
                f"- Phase: {site_found.get('latitude', 'N/A')}\n"
                f"- Company Size: {site_found.get('address_2', 'N/A')}"
            )
        return f"No relevant site found for '{site_query}' within the provided data."

    @kernel_function(
        name="search_sites",
        description="Searches and lists multiple sites based on a search query, like 'sites in Bangalore' or 'active sites'. Returns a Markdown list of site names, IDs, and statuses.",
    )
    def search_sites(self, query: str) -> str:
        sites = self._all_data.get("data", [])
        filtered_sites = search_json_objects(sites, query)

        if not filtered_sites:
            return f"No sites found matching the criteria: '{query}'."

        output = "### Found Sites:\n\n"
        for site in filtered_sites:
            output += (
                f"- **{site.get('location_name', 'N/A')}** (ID: {site.get('site_id', 'N/A')}) - Status: {site.get('state', 'N/A')}\n"
            )
        return output

    @kernel_function(
        name="get_task_details",
        description="Given a task ID or description, retrieve detailed information for that single task. Returns task details in Markdown.",
    )
    def get_task_details(self, task_query: str) -> str:
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

        task_found = None
        for task in filtered_tasks:
            if task.get("task_id", "").lower() == task_query.lower() or task.get("description", "").lower() == task_query.lower():
                task_found = task
                break
        if not task_found and filtered_tasks:
            task_found = filtered_tasks[0]

        if task_found:
            return json.dumps(task_found, indent=2)
        return f"No relevant task found for '{task_query}'."

    @kernel_function(
        name="get_tasks_for_site",
        description="Given a site ID or name, list all associated tasks for that site. Returns a Markdown list of task IDs and descriptions.",
    )
    def get_tasks_for_site(self, site_query: str) -> str:
        sites = self._all_data.get("data", [])
        
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
        return json.dumps(self._all_data, indent=2)
