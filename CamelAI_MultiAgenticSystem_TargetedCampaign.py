# INSTALLATION OF REQUIRED PYTHON PACKAGES
!pip install 'camel-ai[all]'

#DEFINING THE API KEYS
import os
os.environ['OPENAI_API_KEY'] = ''
os.environ['GOOGLE_API_KEY'] =''
os.environ['TAVILY_API_KEY']=''

#IMPORTING THE NECESSARY LIBRARIES
from camel.agents.chat_agent import ChatAgent
from camel.messages.base import BaseMessage
from camel.models import ModelFactory
from camel.societies.workforce import Workforce
from camel.tasks.task import Task
from camel.toolkits import (
    FunctionTool,
    GoogleMapsToolkit,
    SearchToolkit,
)
from camel.types import ModelPlatformType, ModelType

import nest_asyncio
nest_asyncio.apply()

#IMPLEMENTATION OF AGENTS, TASKS AND WORKFORCE
def main():    

    #Define the Model for the Agent as well. Default model is "gpt-4o-mini" and model platform type is OpenAI
    coffee_guide_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )  
    #Define the Tour Guide Agent with the pre-defined model and Google Maps Tool and Prompt
    coffee_guide_agent = ChatAgent(
        BaseMessage.make_assistant_message(
            role_name="Cafe Specialist",
            content="You are a Cafe Specialist",
        ),
        model=coffee_guide_agent_model,
        tools=GoogleMapsToolkit().get_tools()
    )
    
    #Define the web search tool for the Agent using Tavily (we need to define the Tavily API Key beforehand)
    search_toolkit = SearchToolkit()
    search_tools = [
        FunctionTool(search_toolkit.tavily_search)]
    
    #Define the Model for the Agent as well. Default model is "gpt-4o-mini" and model platform type is OpenAI
    search_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT) 
    
    #Define the Search Agent with the pre-defined model and tools and Prompt
    coffee_craft_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Web searching agent",
            content="You can CRAFT PROMOTIONAL CAMPAIGNs SPECIFICALLY FOR each of the CAFEs Based on its unique features",
        ),
        model=search_agent_model,
        tools=search_tools,
    )
    
    #Define the workforce that can take case of multiple agents
    workforce = Workforce('A Cafe Recommender')
    workforce.add_single_agent_worker(
        "Cafe Specialist",
        worker=coffee_guide_agent).add_single_agent_worker(
        "Web searching agent",
        worker=coffee_craft_agent)

    # specify the task to be solved Defining the exact task needed
    human_task = Task(
        content=(
            "Tell me about 2 major coffee shops with their details in Manhattan along with their locations and price of Cappuccino there. Also craft a PROMOTIONAL CAMPAIGN SPECIFICALLY FOR each of THE CAFEs Based on its unique features."
        ),
        id='0',
    )
    task = workforce.process_task(human_task)
    print('Final Result of Original task:\n', task.result)

# EXECUTING THE FUNCTION & PRINTING THE OUTPUT
print(main())
