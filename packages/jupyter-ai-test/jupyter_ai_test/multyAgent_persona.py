from typing import Any
from jupyter_ai.personas.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message
from jupyter_ai.history import YChatHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from agno.agent import Agent
from agno.models.aws import AwsBedrock, Claude
import boto3
from agno.team.team import Team
from agno.tools.python import PythonTools
from agno.tools.file import FileTools
from agno.tools.github import GithubTools

from .multyAgent_templates import MultyAgentVariables , _MULTYAGENT_PROMPT_TEMPLATE
session = boto3.Session()

from langchain_core.runnables import Runnable

class TeamCoordinator(Runnable):
    """Agent that coordinates a team by delegating to the team's run method"""
    
    def __init__(self, team: Team, **kwargs):
        self.team = team
        self.kwargs = kwargs
        
    def invoke(self, input: Any) -> Any:
        """Invoke the team with the given input"""
        return self.team.run(input)

class MultyAgentPersona(BasePersona):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self):
        return PersonaDefaults(
            name="MultyAgentPersona",
            avatar_path="/api/ai/static/jupyternaut.svg",
            description="A specialized agent for Jupyter notebook cells with command-based functionality.",
            system_prompt="I am a multi-agent system designed to help with coding tasks in Jupyter notebooks. I coordinate a team of specialized agents: a planner who breaks down tasks into clear steps, a coder who implements solutions following best practices, a tester who ensures code quality through comprehensive testing, and a GitHub specialist who manages repository operations. Together, we can help you with planning, implementing, testing, and managing your code in GitHub.",
        )
    
    async def process_message(self, message: Message):
        provider_name = self.config.lm_provider.name
        model_id = self.config.lm_provider_params["model_id"]

        # Create variables for the template
        variables = MultyAgentVariables(
            input=message.body,
            model_id=model_id,
            provider_name=provider_name,
            persona_name=self.name
        )

        # Create team members
        planner = Agent(
            name="planner",
            role="Strategic planner who breaks down tasks into clear, actionable steps",
            model=Claude(id=model_id, session=session),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Analyze user requests and break them down into clear, manageable steps",
                "Consider technical requirements, dependencies, and potential challenges"
            ],
            markdown=True
        )

        coder = Agent(
            name="coder",
            role="Expert programmer responsible for implementing solutions",
            model=Claude(id=model_id, session=session),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Implement code following the planner's specifications",
                "Write clean, efficient, and well-documented code",
                "Follow Python best practices and PEP 8 style guidelines"
            ],
            tools=[PythonTools()],
            markdown=True
        )

        tester = Agent(
            name="tester",
            role="Quality assurance engineer focused on testing and validation",
            model=Claude(id=model_id, session=session),
            instructions=[
                "Do not create new files unless explicitly asked by user.",
                "Only write and run tests when explicitly requested by the user",
                "When testing, ensure coverage for both normal cases and edge cases",
                "When testing, include tests for error conditions and invalid inputs",
                "Follow testing best practices and naming conventions",
                "Document test cases and their purpose clearly"
            ],
            tools=[PythonTools()],
            markdown=True
        )

        github = Agent(
            name="gitHub",
            role="GitHub operations specialist",
            model=Claude(id=model_id, session=session),
            instructions=[
                "Monitor and analyze GitHub repository activities and changes",
                "Help with repository organization and maintenance",
                "Ensure proper Git workflow practices are followed",
                "Handle branch management and merging strategies",
                "Provide insights on repository metrics and activity patterns"
            ],
            tools=[GithubTools()],
            markdown=True
        )

        file_manager = Agent(
            name="fileManager",
            role="File operations manager",
            model=Claude(id=model_id, session=session),
            instructions=[
                "Assist with local file management",
                "Only read a file when explicitly requested",
                "Only write to a file when explicitly requested"
            ],
            tools=[FileTools()],
            markdown=True
        )

        # Create team
        team = Team(
            name="dev-team",
            mode="coordinate",
            members=[planner, coder, tester, github, file_manager],
            model=Claude(id=model_id, session=session),
            instructions=[
                "Coordinate between planner, coder, tester, and GitHub specialist to deliver high-quality solutions",
                "Do not attempt to write test cases or test the code unless explicitly asked by user",
                "Do not run any tests unless explicitly requested by user",
                "Do not create new files unless explicitly asked by user",
                "Ensure smooth handoffs between planning, implementation, testing, and repository management phases",
                "Maintain clear communication between team members",
                "Validate that all requirements are met in the final solution",
                "Ensure code quality standards are maintained throughout the development process",
                "Address any conflicts or inconsistencies between different phases",
                "Facilitate collaboration through proper Git workflow and code review processes"
            ],
            markdown=True,
            show_members_responses=True,
            enable_agentic_context=True,
            add_datetime_to_instructions=True
        )

        # Create runnable chain with history
        runnable = RunnableWithMessageHistory(
            runnable=_MULTYAGENT_PROMPT_TEMPLATE | TeamCoordinator(
                team=team,
                name="team-coordinator",
                role="Team coordinator that manages the development team",
                model=Claude(id=model_id, session=session),
                instructions=[
                    "Coordinate the development team to deliver high-quality solutions",
                    "Pass requests to the team and return their responses"
                ],
                markdown=True
            ),
            get_session_history=lambda: YChatHistory(ychat=self.ychat, k=2),
            input_messages_key="input",
            history_messages_key="history"
        )

        response_stream = runnable.astream(variables.model_dump())
        await self.stream_message(response_stream)
