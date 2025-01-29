!pip install crewai crewai-tools poetry
!pip install langchain_openai


from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import VisionTool
import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI


os.environ['OPENAI_API_KEY'] =''
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o-mini"
image_url = "https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQQ8NVc0s_aQ-y97-jJlBpGQyHG8cbu1_Gr2i7htKhpE5giFJVy-ZSyaUcpXo7ExvF2H7aocQziYBqUGUx0EKNCVniAc0I37Rble0wFIu4HDah3hDrcfwryDKvS&usqp=CAE"
vision_tool = VisionTool()
llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0.1,
    max_tokens=8000
)

image_text_extractor = Agent(
     role='Item Name & Description Extraction Specialist',
     goal="Extract NAME OF ITEM PRESENT ALONG WITH THEIR DESCRIPTION from images efficiently using AI-powered tools. You should get ITEM NAMES  from %s"%image_url,
     backstory='You are an expert in NAME OF ITEM PRESENT ALONG WITH THEIR DESCRIPTION extraction, specializing in using AI to process. Make sure you use the tools provided.',
      tools=[vision_tool],allow_delegation=False,verbose=True)

def text_extraction_task(agent):
        return Task(
            description = """Extract NAME OF ITEM PRESENT ALONG WITH THEIR DESCRIPTION from the provided image file. Ensure that the ITEM NAME & DESCRIPTION  is accurate and complete,
    and ready for any further analysis or processing tasks. The image file provided may contain
    various products of Different BRANDS, so it's crucial to capture all readable text. """,
            agent = agent,
            expected_output = "A string containing NAME OF ITEM PRESENT ALONG WITH THEIR DESCRIPTION extracted from the image.",
           max_iter=1
        )

description_generator = Agent(
     role='Crafting Specialist',
     goal='From  the item names & description extracted from the previous agent, craft a good description of the PRODUCT (not any PERSON) highlighting all its key features for displaying on a website',
     backstory='You are an expert in crafting good descriptions for displaying on websites',
      llm=llm,allow_delegation=False,verbose=True)
def description_generator_task(agent):
        return Task(
            description = "From  the item names & description extracted from the previous agent, craft a good description of the PRODUCT (not any PERSON) highlighting all its key features for displaying on a website",
            agent = agent,
            expected_output = "A string containing a good description of the product.",
         max_iter=1)


title_generator = Agent(
     role='Item Title Specialist',
     goal='From  the item description crafted from the previous agent, craft a good title for the PRODUCT (not any PERSON) in maximum 3 words for displaying on a ecommerce website',
     backstory='You are an expert in creating eye catching titles for displaying on websites',
      llm=llm,allow_delegation=False,verbose=True)
def title_generator_task(agent):
        return Task(
            description = "From  the item description crafted from the previous agent, ADD to the Description of the Product generated from previous agent A GOOD TITLE for the PRODUCT (not any PERSON) in maximum 3 words for displaying on a ecommerce website. Output should be Description of the Product generated from previous agent along with the Title",
            agent = agent,
            expected_output = "Output should be Description of the Product generated from previous agent along with the Title",
              max_iter=1)



task1 = text_extraction_task(image_text_extractor)
task2 = description_generator_task(description_generator)
task3 = title_generator_task(title_generator)


#start crew
targetting_crew = Crew(
    agents=[image_text_extractor,description_generator,title_generator],
    tasks=[task1,task2,task3],
    verbose=True, 
  process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
  )
targetting_result = targetting_crew.kickoff()



print(targetting_result)


