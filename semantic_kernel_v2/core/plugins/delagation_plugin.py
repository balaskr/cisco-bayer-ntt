# Semantic Kernel Imports
from semantic_kernel.functions import kernel_function

from core.agents.sites_tasks_agent import SitesTasksAgent

import json

class DelegationPlugin:
    """
    A plugin to interact with other specialized sub-agents.
    This enables a higher-level Manager Agent to delegate tasks.
    """

    @kernel_function(
        name="call_sites_tasks_agent",
        description="Delegates a query to the Sites & Tasks Agent to retrieve information about sites and their associated tasks. Use this for any query specifically related to site details, searching sites, task details, or tasks for a specific site.",
    )
    async def call_sites_tasks_agent(self, query: str) -> str:
        """
        Calls the main query function of the Sites & Tasks Agent.

        Args:
            query (str): The user's query relevant to sites or tasks.

        Returns:
            str: The response from the Sites & Tasks Agent.
        """
        print(f"\nDelegating query '{query}' to Sites & Tasks Agent...")
        try:
            # this should be an api call in prod
            context = await self.simulate_api_call()
            sites_tasks_agent = SitesTasksAgent(context)
            agent_response = await sites_tasks_agent.run(query)
            
            return f"Sites & Tasks Agent responded:\n{agent_response}"
        except Exception as e:
            error_message = f"Error delegating to Sites & Tasks Agent: {e}"
            print(f"ERROR: {error_message}")
            return error_message

    async def simulate_api_call(self,path="knowledge/data.json"):
        with open(path, 'r', encoding='utf-8') as f:
            global_site_tasks_data = json.load(f)
        return global_site_tasks_data
