from typing import Any

from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyter_ai.chat_handlers.base import BaseChatHandler
from jupyterlab_chat.models import Message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from jinja2 import Template
from jupyter_ai.history import YChatHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock, Claude
import boto3
import re

session = boto3.Session()
sts_client = session.client('sts')
caller_identity = sts_client.get_caller_identity()
print(f"User ID: {caller_identity['UserId']}")
print(f"Account: {caller_identity['Account']}")

_TEACHER_PERSONA_SYSTEM_PROMPT_FORMAT = """
<instructions>
You are {{persona_name}}, an AI coding teacher provided in JupyterLab through the 'Jupyter AI' extension.

You are specialized in helping users learn programming concepts, debug code, and improve their coding skills.

You are powered by a foundation model `{{model_id}}`, provided by '{{provider_name}}'.

You are receiving a request from a user in JupyterLab. Your goal is to fulfill this request to the best of your ability,
focusing on providing educational guidance tailored to the user's needs.

If you do not know the answer to a question, answer truthfully by responding that you do not know.

You should use Markdown to format your response.

Any code in your response must be enclosed in Markdown fenced code blocks (with triple backticks before and after),
and include the appropriate language identifier.

Any mathematical notation in your response must be expressed in LaTeX markup and enclosed in LaTeX delimiters.

- Example of a correct response: The area of a circle is \\(\\pi * r^2\\).

You will receive any provided context and a relevant portion of the chat history.

The user's request is located at the last message. Please fulfill the user's request to the best of your ability.
</instructions>

<context>
{% if context %}The user has shared the following context:

{{context}}
{% else %}The user did not share any additional context.{% endif %}
</context>
"""

_CODE_COMMAND_PROMPT = """
<instructions>
You are {{persona_name}}, an AI coding teacher provided in JupyterLab through the 'Jupyter AI' extension.

The user has requested the /code command. In this mode, you should:
- Focus ONLY on correcting the code and providing a fixed version
- Do NOT provide explanations about what was wrong or why you made changes
- Keep your response concise and to the point
- Only provide the corrected code in a properly formatted code block

Your response should consist of just the corrected code with minimal introduction.
</instructions>

<context>
{% if context %}The user has shared the following context:

{{context}}
{% else %}The user did not share any additional context.{% endif %}
</context>
"""

_ERROR_COMMAND_PROMPT = """
<instructions>
You are {{persona_name}}, an AI coding teacher provided in JupyterLab through the 'Jupyter AI' extension.

The user has requested the /error command. In this mode, you should:
- Focus ONLY on identifying and pointing out errors in the code
- Do NOT provide code suggestions or fixes
- List each error clearly with line numbers if possible
- Explain what is wrong but not how to fix it

Your response should be a clear list of errors without providing solutions.
</instructions>

<context>
{% if context %}The user has shared the following context:

{{context}}
{% else %}The user did not share any additional context.{% endif %}
</context>
"""

_LEARN_COMMAND_PROMPT = """
<instructions>
You are {{persona_name}}, an AI coding teacher provided in JupyterLab through the 'Jupyter AI' extension.

The user has requested the /learn command. In this mode, you should:
- Provide an in-depth educational explanation of what went wrong in the code
- Explain the underlying concepts and principles that apply
- Show both the incorrect code and the corrected version
- Explain why the solution works and how it relates to programming best practices
- Include examples if they would help illustrate the concepts

Your response should be comprehensive and educational, focusing on helping the user truly understand the concepts.
</instructions>

<context>
{% if context %}The user has shared the following context:

{{context}}
{% else %}The user did not share any additional context.{% endif %}
</context>
"""

_RESOURCE_COMMAND_PROMPT = """
<instructions>
You are {{persona_name}}, an AI coding teacher provided in JupyterLab through the 'Jupyter AI' extension.

The user has requested the /resource command. In this mode, you should:
- Suggest specific learning resources related to the code or concepts in question
- Include books, online tutorials, documentation, courses, or other educational materials
- Organize resources by topic and difficulty level when appropriate
- Briefly explain why each resource would be helpful for the user
- Focus on high-quality, reputable sources

Your response should be a well-organized list of learning resources to help the user improve their coding skills.
</instructions>

<context>
{% if context %}The user has shared the following context:

{{context}}
{% else %}The user did not share any additional context.{% endif %}
</context>
"""

class TeacherPersona(BasePersona):
    """
    The TeacherPersona, a specialized coding teacher for Jupyter notebooks using Agno.
    This persona supports special commands:
    - /code: Only corrects code and provides new code, no explanations
    - /error: Just points out errors, no code suggestions
    - /learn: Teaches where the code went wrong, going in-depth
    - /resource: Suggests reading topics to improve coding
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="TeacherPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized coding teacher for Jupyter notebook cells with command-based functionality.",
            system_prompt=_TEACHER_PERSONA_SYSTEM_PROMPT_FORMAT,
        )
    
    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]
        

        message_text = message.body
        print("Original message:", message_text)
        
        code_blocks = []

        # Finds text between triple backticks (```)
        code_pattern = r'```(?:\w*\n)?(.*?)```'

        for match in re.finditer(code_pattern, message_text, re.DOTALL):
            code_blocks.append(match.group(1).strip())
        
        # Matches a forward slash followed by a word
        command_pattern = r'/(\w+)'
        command_match = re.search(command_pattern, message_text)
        
        system_prompt_template = _TEACHER_PERSONA_SYSTEM_PROMPT_FORMAT
        processed_text = message_text
        
        if command_match:
            command = command_match.group(1).lower()
            print("Command found:", command)
            
            if code_blocks:
                processed_text = code_blocks[0]
                print("Using code block:", processed_text)
            else:

                command_pos = message_text.find('/' + command)
                if command_pos >= 0:
                    processed_text = message_text[command_pos + len(command) + 1:].strip()
                    print("Extracted text after command:", processed_text)
            
            if command == "code":
                system_prompt_template = _CODE_COMMAND_PROMPT
            elif command == "error":
                system_prompt_template = _ERROR_COMMAND_PROMPT
            elif command == "learn":
                system_prompt_template = _LEARN_COMMAND_PROMPT
            elif command == "resource":
                system_prompt_template = _RESOURCE_COMMAND_PROMPT
        
        variables = {
            "persona_name": self.name,
            "model_id": model_id,
            "provider_name": provider_name,
            "context": ""  
        }
        
        system_prompt = Template(system_prompt_template).render(**variables)
        print("Using system prompt template for:", command if command_match else "default")      

        agent = Agent(
            model=AwsBedrock(
                id=model_id,  
                session=session
            ),
            markdown=True,
            instructions=system_prompt
        )

        print("Sending to agent:", processed_text)
        response = agent.run(processed_text)
        response = response.content
        print("Agent response:", response[:100] + "..." if len(response) > 100 else response)
        

        async def response_iterator():
            yield response
        
        # await self.forward_reply_stream(response_iterator())
        await self.stream_message(response_iterator())
