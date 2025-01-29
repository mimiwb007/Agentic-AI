!pip install "griptape[all]" -U

from duckduckgo_search import DDGS
from griptape.artifacts import TextArtifact
from griptape.drivers import LocalStructureRunDriver
from griptape.rules import Rule
from griptape.structures import Agent, Pipeline, Workflow
from griptape.tasks import CodeExecutionTask, PromptTask, StructureRunTask

from griptape.drivers import GoogleWebSearchDriver, LocalStructureRunDriver
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent, Workflow
from griptape.tasks import PromptTask, StructureRunTask
from griptape.tools import (
    PromptSummaryTool,
    WebScraperTool,
    WebSearchTool,
)
from griptape.drivers import DuckDuckGoWebSearchDriver
import os
os.environ["OPENAI_API_KEY"]=''

#Defining the Writers
WRITERS = [
    {
        "role": "Luxury Blogger",
        "goal": "Inspire Luxury with stories of royal things that people aspire",
        "backstory": "You bring aspirational and luxurious things to your audience through vivid storytelling and personal anecdotes.",
    },
    {
        "role": "Lifestyle Freelance Writer",
        "goal": "Share practical advice on living a balanced and stylish life",
        "backstory": "From the latest trends in home decor to tips for wellness, your articles help readers create a life that feels both aspirational and attainable.",
    },
]

#Defining the Researcher Agent
def build_researcher() -> Agent:
    """Builds a Researcher Structure."""
    return Agent(
        id="researcher",
        tools=[
            WebSearchTool(
                web_search_driver=DuckDuckGoWebSearchDriver(),
            ),
            WebScraperTool(
                off_prompt=True,
            ),
            PromptSummaryTool(off_prompt=False),
        ],
        rulesets=[
            Ruleset(
                name="Position",
                rules=[
                    Rule(
                        value="Lead Real Estate Analyst",
                    )
                ],
            ),
            Ruleset(
                name="Objective",
                rules=[
                    Rule(
                        value="Discover Real Estate advancements in and around Delhi NCR",
                    )
                ],
            ),
            Ruleset(
                name="Background",
                rules=[
                    Rule(
                        value="""You are part of a Real Estate Brokering Company.
                        Your speciality is spotting new trends in Real Estate for buyers and sells.
                        You excel at analyzing intricate data and delivering practical insights."""
                    )
                ],
            ),
            Ruleset(
                name="Desired Outcome",
                rules=[
                    Rule(
                        value="Comprehensive analysis report in list format",
                    )
                ],
            ),
        ],
    )

#Defining the writer agent
def build_writer(role: str, goal: str, backstory: str) -> Agent:
    """Builds a Writer Structure.

    Args:
        role: The role of the writer.
        goal: The goal of the writer.
        backstory: The backstory of the writer.
    """
    return Agent(
        id=role.lower().replace(" ", "_"),
        rulesets=[
            Ruleset(
                name="Position",
                rules=[
                    Rule(
                        value=role,
                    )
                ],
            ),
            Ruleset(
                name="Objective",
                rules=[
                    Rule(
                        value=goal,
                    )
                ],
            ),
            Ruleset(
                name="Backstory",
                rules=[Rule(value=backstory)],
            ),
            Ruleset(
                name="Desired Outcome",
                rules=[
                    Rule(
                        value="Full blog post of at least 4 paragraphs",
                    )
                ],
            ),
        ],
    )

#Defining Tasks
team = Workflow()
research_task = team.add_task(
        StructureRunTask(
            (
                """Perform a detailed examination of the newest developments in Real Estate Updates in gurgaon as of 2025.
                Pinpoint major trends, new upcoming properties and any projections.""",
            ),
            id="research",
            structure_run_driver=LocalStructureRunDriver(
                create_structure=build_researcher,
            ),
        ),
    )

writer_tasks = team.add_tasks(
        *[
            StructureRunTask(
                (
                    """Using insights provided, develop an engaging blog
                post that highlights the most significant real estate updates of Gurgaon.
                Your post should be informative yet accessible, catering to a general audience.
                Make it sound cool, avoid complex words.

                Insights:
                {{ parent_outputs["research"] }}""",
                ),
                structure_run_driver=LocalStructureRunDriver(
                    create_structure=lambda writer=writer: build_writer(
                        role=writer["role"],
                        goal=writer["goal"],
                        backstory=writer["backstory"],
                    )
                ),
                parent_ids=[research_task.id],
            )
            for writer in WRITERS
        ]
    )

end_task = team.add_task(
        PromptTask(
            'State "All Done!"',
            parent_ids=[writer_task.id for writer_task in writer_tasks],
        )
    )

#Executing the Task
team.run()



