!pip install 'camel-ai[all]'

import os
os.environ['OPENAI_API_KEY'] = ''

# IMPORTING NECESSARY LIBRARIES
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
from camel.toolkits import DalleToolkit
from camel.types import ModelPlatformType, ModelType
import nest_asyncio
nest_asyncio.apply()


#DEFINING THE AGENTS
#Define the Model for the Agent as well. Default model is "gpt-4o-mini" and model platform type is OpenAI
guide_agent_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
    )  

#Generate A Story For 1 Year Old Kids on a Topic
real_estate_agent = ChatAgent(
        BaseMessage.make_assistant_message(
            role_name="Real Estate Specialist",
            content="You are a Real Estate Specialist who is an expert in creating Description of Upcoming Residential Projects",
        ),
        model=guide_agent_model,
    )

#Generate A Story For 1 Year Old Kids on a Topic
property_title_agent = ChatAgent(
        BaseMessage.make_assistant_message(
            role_name="Real Estate Project Name Specialist",
            content="You are a Real Estate Project Name Specialist who is an expert in Generating Trendy Names FoR Residental Projects in india",
        ),
        model=guide_agent_model,
    )
    
#Define the web search tool for the Agent using Tavily (we need to define the Tavily API Key beforehand)
dalletool = DalleToolkit()
imagegen_tools = [
    FunctionTool(dalletool.get_dalle_img),
    
]

 #Define the Image Generation Agent with the pre-defined model and tools and Prompt
 image_generation_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Image Generation Specialist",
            content="You can Generate Images For Upcoming Real Estate Projects For Showing to Clients",
        ),
        model=guide_agent_model,
        tools=imagegen_tools,
    )



#DEFINE THE WORKFORCE THAT CAN TAKE CASE OF MULTIPLE AGENTS
workforce = Workforce('Real Estate Brochure Generator')
workforce.add_single_agent_worker(
        "Real Estate Specialist",
        worker=real_estate_agent).add_single_agent_worker(
        "Real Estate Project Name Specialist",
        worker=property_title_agent).add_single_agent_worker(
        "Image Generation Specialist",
        worker=image_generation_agent)

 # specify the task to be solved Defining the exact task needed
 human_task = Task(
        content=(
            """Craft a Brochure Content For a Upcoming Residential Real Estate Project in Gurgaon. 
            Provide a Name for this Property as well.
            Generate an Image of this Upcoming Project as well."""
        ),
        id='0',
    )
task = workforce.process_task(human_task)

print(task)
    
