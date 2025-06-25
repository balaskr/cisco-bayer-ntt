import asyncio
import json
import logging
import os
from typing import Any, Dict

import azure.functions as func
import requests  # For fetching data from the external API

from core.agents import get_agents
from core.plugins import SiteTasksPlugin
# Semantic Kernel Imports
from semantic_kernel.agents import HandoffOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import (AuthorRole, ChatMessageContent,
                                      FunctionCallContent,
                                      FunctionResultContent)
from semantic_kernel.kernel import Kernel
from semantic_kernel.utils.logging import setup_logging

from dotenv import load_dotenv;load_dotenv()


# Configure Semantic Kernel logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.INFO)

# Global variable for the Azure Function app instance
app = func.FunctionApp()

agent_response: str = None

def agent_response_callback(message: ChatMessageContent) -> None:
    global agent_response
    logging.info(f"\n--- Agent Response ({message.name}) ---")
    if message.content:
        logging.info(f"Content: {message.content}")
        agent_response = str(message.content)
    # for item in message.items:
    #     if isinstance(item, FunctionCallContent):
    #         logging.info(f"Function Call: '{item.name}' with arguments '{item.arguments}'")
    #     if isinstance(item, FunctionResultContent):
    #         result_str = str(item.result)
    #         if len(result_str) > 500:
    #             logging.info(f"Function Result from '{item.name}':\n{result_str[:500]}...\n(Truncated for display)")
    #         else:
    #             logging.info(f"Function Result from '{item.name}':\n{result_str}")
    logging.info("-------------------------------------")


# This function is not used in the HTTP trigger, but kept for completeness of the original SK code.
async def human_response_function() -> ChatMessageContent:
    user_input = "Agent requires input. This won't be used in HTTP trigger."
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


async def run_semantic_kernel_agent_query(user_query: str, site_tasks_data: Dict[str, Any]) -> str:
    """
    Main function to run the Semantic Kernel handoff orchestration with provided data.
    """
    logging.info(f"Initiating Semantic Kernel with query: '{user_query}'")

    kernel = Kernel()

    data_plugin_instance = SiteTasksPlugin(site_tasks_data)

    agents, handoffs = get_agents(kernel, data_plugin_instance)
    handoff_orchestration = HandoffOrchestration(
        members=agents,
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
    )

    runtime = InProcessRuntime() 
    runtime.start()

    orchestration_result = await handoff_orchestration.invoke(
        task=user_query,
        runtime=runtime,
    )

    final_result_content = ""
    value = await orchestration_result.get()
    if isinstance(value, ChatMessageContent):
        final_result_content = value.content
    elif isinstance(value, str):
        final_result_content = value
    else:
        final_result_content = str(value)
    
    logging.info(f"\n--- Final Orchestration Result ---")
    logging.info(final_result_content)

    await runtime.stop_when_idle()
    logging.info("Runtime stopped.")
    
    return final_result_content

# --- Azure Function HTTP Trigger Entry Point ---

@app.function_name(name="skAgenticPoCFunc")
@app.route(route="skAgenticPoCFunc", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
async def skAgenticPoCFunc(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON body. Please provide a JSON payload.",
            status_code=400
        )
    
    # Extract parameters from the JSON body
    user_input = req_body.get("text")
    # user_id = req_body.get("from", {}).get("id") # Not directly used by SK but often useful for logging/context
    # chat_interaction_id = req_body.get("conversation", {}).get("id") # Not directly used by SK but often useful for logging/context
    # auth_header = req_body.get("channelData", {}).get("requestHeader", {}).get("X-Insight-Token")

    # Validate required parameters for this specific function
    if not user_input:
        return func.HttpResponse(
            "Missing required parameter: 'text' in the JSON body.",
            status_code=400
        )
    # if not auth_header:
    #     return func.HttpResponse(
    #         "Missing required parameter: 'X-Insight-Token' in channelData.requestHeader.",
    #         status_code=400
    #     )

    try:
        # 1. Fetch data from the external API using the provided URL and X-Insight-Token
        # data_api_url = "https://int.portal.nttltd.global.ntt/l/tis/project-overview-api/v1/task/sitesanalysis?page=1&limit=100"
        # headers = {
        #     "Accept": "*/*",
        #     "X-Insight-Token": auth_header
        # }
        
        # logging.info(f"Attempting to fetch data from: {data_api_url}")
        # api_response = requests.get(data_api_url, headers=headers, timeout=120)
        # api_response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        site_tasks_data = json.load(open("knowledge/data.json"))
        logging.info("Successfully fetched data from external API.")

        # 2. Invoke the Semantic Kernel multi-agent system
        final_response_content = await run_semantic_kernel_agent_query(user_input, site_tasks_data)
        print(agent_response)
        # 3. Return the Semantic Kernel's final response
        return func.HttpResponse(
            json.dumps({"response": agent_response}),
            mimetype="application/json",
            status_code=200
        )

    except requests.exceptions.Timeout:
        logging.error("API call to data source timed out.")
        return func.HttpResponse(
            json.dumps({"error": "Failed to fetch data: The request to the data source timed out. Please try again."}),
            mimetype="application/json",
            status_code=504 # Gateway Timeout
        )
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error fetching data from API: {e.response.status_code} - {e.response.text}")
        return func.HttpResponse(
            json.dumps({"error": f"Failed to fetch data from external API: {e.response.status_code} {e.response.reason}. Details: {e.response.text}"}),
            mimetype="application/json",
            status_code=e.response.status_code
        )
    except requests.exceptions.RequestException as e:
        logging.error(f"Network or request error fetching data from API: {e}")
        return func.HttpResponse(
            json.dumps({"error": f"Network error connecting to data source: {str(e)}. Please check connectivity."}),
            mimetype="application/json",
            status_code=503 # Service Unavailable
        )
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from external API response.")
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON response received from the data source API."}),
            mimetype="application/json",
            status_code=500
        )
    except Exception as e:
        logging.exception("An unexpected error occurred during function execution.") # logs traceback
        return func.HttpResponse(
            json.dumps({"error": f"An unexpected server error occurred: {str(e)}. Please check logs for details."}),
            mimetype="application/json",
            status_code=500
        )