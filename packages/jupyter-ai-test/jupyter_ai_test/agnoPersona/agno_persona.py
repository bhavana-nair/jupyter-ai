from typing import Any

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyter_ai.chat_handlers.base import BaseChatHandler
from jupyterlab_chat.models import Message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from jinja2 import Template
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock, Claude
import boto3

from .agnoPersona_templates import AGNO_PROMPT_TEMPLATE, AgnoPersonaVariables

session = boto3.Session()

class AgnoPersona(BasePersona):
    """
    The AgnoPersona, a specialized coding assistant for Jupyter notebooks using Agno.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="AgnoPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized coding assistant for Jupyter notebook cells with Agno integration.",
            system_prompt="I am an AI coding assistant powered by Agno, designed to help with coding tasks in Jupyter notebooks. I can help with coding, debugging, and providing explanations about code. I maintain chat history to provide more contextual and relevant responses.",
        )
    
    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]

        # Create variables for the template
        variables = AgnoPersonaVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name
        )

        runnable = RunnableWithMessageHistory(
            runnable=AGNO_PROMPT_TEMPLATE | Agent(
                model=AwsBedrock(
                    id=model_id,
                    session=session
                ),
                markdown=True
            ),
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=2),
            input_messages_key="input",
            history_messages_key="history"
        )

        response_stream = runnable.astream(variables.model_dump())
        await self.stream_message(response_stream)
