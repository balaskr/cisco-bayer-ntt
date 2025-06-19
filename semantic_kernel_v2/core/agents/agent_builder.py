from semantic_kernel.agents import ChatCompletionAgent

from semantic_kernel.contents import (AuthorRole, ChatMessageContent,
                                      FunctionCallContent,
                                      FunctionResultContent)


class AgentBuilder(ChatCompletionAgent):
    
    def __init__(
            self,
            name:str,
            description:str,
            instructions:str,
            service: ChatCompletionAgent,
            plugins,
    ):  
        super().__init__(
            name=name,
            description=description,
            instructions=instructions,
            service=service,
            plugins=plugins)
    
    async def run(self,user_query: str) -> str:
        _last_response:str = None

        async for response in self.invoke(
            messages=[ChatMessageContent(role=AuthorRole.USER, content=user_query)],
            on_intermediate_message=self._handle_intermediate_steps
        ):
            _last_response = response.content
        
        return _last_response

    
    async def _handle_intermediate_steps(self, msg: ChatMessageContent) -> None:
        if any(isinstance(item, FunctionResultContent) for item in msg.items):
            for fr in msg.items:
                if isinstance(fr, FunctionResultContent):
                    print(f"Function Result:> ....... for function: {fr.name}")
        elif any(isinstance(item, FunctionCallContent) for item in msg.items):
            for fcc in msg.items:
                if isinstance(fcc, FunctionCallContent):
                    print(f"Function Call:> {fcc.name} with arguments: {fcc.arguments}")

