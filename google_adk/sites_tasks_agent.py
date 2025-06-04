import asyncio
from google.adk.agents import Agent, BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types # For creating message Content/Parts
from typing import AsyncGenerator
from google_adk.utils import prompts, mock_api

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(
    level=logging.INFO,  # or DEBUG if you want more verbosity
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

MODEL_GEMINI_2_0_FLASH = "gemini-1.5-flash"

AGENT_MODEL = MODEL_GEMINI_2_0_FLASH

class SitesTasksAgent(BaseAgent):
    """
    Custom agent for helping with sites and tasks.

    This agent orchestrates a sequence of LLM agents to generate various informations
    about sites, tasks and executive summaries.
    """
    delegator: LlmAgent
    sites_helper: LlmAgent
    tasks_helper: LlmAgent
    overall_helper: LlmAgent

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
            self,
            name:str,
            delegator:LlmAgent,
            sites_helper: LlmAgent,
            tasks_helper: LlmAgent,
            overall_helper: LlmAgent,
    ):
        """
        Initializes the SitesTasksAgent.

        Args:
              name: The name of the agent.
              delegator: An LlmAgent to recognize user intent and decide the subsequent agent call.
              sites_helper: An LlmAgent to provide assistance with sites information.
              tasks_helper: An LlmAgent to provide assistance with tasks information.
              overall_helper: An LlmAgent to provide summaries of the data.
        """  

        sub_agents_list = [
            delegator,
            sites_helper,
            tasks_helper,
            overall_helper
        ]
        
        super().__init__(
            name=name,
            delegator=delegator,
            sites_helper= sites_helper,
            tasks_helper=tasks_helper,
            overall_helper=overall_helper,
        )


    async def _run_async_impl(
            self, ctx:InvocationContext
        ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for SitesTasksAgent.
        """

        logger.info(f"[{self.name}] starting delegation")
        async for event in self.delegator.run_async(ctx):
            logger.info(f"[{self.name}] Event from Delegator: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event

        logger.info("Intent: "+ ctx.session.state.get('intent'))

        intent = ctx.session.state.get('intent')

        if 'SITE' in intent:
            logger.info(f"[{self.name}] rerouting to sites helper")
            async for event in self.sites_helper.run_async(ctx):
                logger.info(f"[{self.name}] Event from SitesHelper: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event

        if 'TASK' in intent:
            logger.info(f"[{self.name}] rerouting to tasks helper")
            async for event in self.tasks_helper.run_async(ctx):
                logger.info(f"[{self.name}] Event from TasksHelper: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event

        if 'OVERALL' in intent:
            logger.info(f"[{self.name}] rerouting to overall helper")
            async for event in self.overall_helper.run_async(ctx):
                logger.info(f"[{self.name}] Event from OverallHelper: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event


delegator = LlmAgent(
    name="delegator",
    model=MODEL_GEMINI_2_0_FLASH,
    instruction=prompts.delegator,
    input_schema=None,
    # tools=[mock_api],
    output_key="intent"
)

sites_helper = LlmAgent(
    name="SitesHelper",
    model=MODEL_GEMINI_2_0_FLASH,
    instruction=prompts.sites,
    input_schema=None
)

tasks_helper = LlmAgent(
    name="SitesHelper",
    model=MODEL_GEMINI_2_0_FLASH,
    instruction="You are the Task Agent. A user is asking about tasks or projects. Respond with a confirmation that you are the Task Agent and acknowledge the request for task information.",
    input_schema=None
)

overall_helper = LlmAgent(
    name="SitesHelper",
    model=MODEL_GEMINI_2_0_FLASH,
    instruction="You are the Overall Agent. A user is asking for a summary or general overview. Respond with a confirmation that you are the Overall Agent and acknowledge the request for an overall summary.",
    input_schema=None
)

sites_tasks_agent = SitesTasksAgent(
    name="SitesTasksAgent",
    delegator=delegator,
    sites_helper=sites_helper,
    tasks_helper=tasks_helper,
    overall_helper=overall_helper
)

APP_NAME = "xyz"
USER_ID = "user01"
SESSION_ID = "session01"


async def main():
    session_service = InMemorySessionService()
    # initial_state = {"sites_context" : mock_api()}
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        #state=initial_state
    )
    logger.info(f"Initial session state: {session.state}")

    runner = Runner(
        agent=sites_tasks_agent,
        app_name=APP_NAME,
        session_service=session_service
    )

    async def call_agent(query: str):
        content = types.Content(role='user', parts=[types.Part(text=f"{query}\n\nJson Data:{mock_api()}")])
        events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

        final_response = "No final response captured."
        for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                logger.info(f"Potential final response from [{event.author}]: {event.content.parts[0].text}")
                final_response = event.content.parts[0].text

        print("\n--- Agent Interaction Result ---")
        print("Agent Final Response: ", final_response)

        final_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        # print("Final Session State:")
        # import json
        # print(json.dumps(final_session.state, indent=2))
        print("-------------------------------\n")

    await call_agent("list my sites in a table.")
    

if __name__ == "__main__":
    asyncio.run(main())
    