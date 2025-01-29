!pip install llama-index-core
!pip install llama-index-readers-file
!pip install llama-index-embeddings-openai
!pip install llama-index-llms-llama-api
!pip install 'crewai[tools]'
!pip install llama-index-llms-langchain
!pip install llama-index-llms-langchain


import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from langchain_openai import ChatOpenAI
os.environ['OPENAI_API_KEY']=""


from crewai_tools import ScrapeWebsiteTool
# To enable scrapping any website it finds during it's execution
tool = ScrapeWebsiteTool()

# Initialize the tool with the website URL, 
# so the agent can only scrap the content of the specified website
tool = ScrapeWebsiteTool(website_url='https://engocontrols.com/en/8-ways-to-reduce-energy-consumption/')


def create_crew(taste):

  #Agent For Creating an Energy Optimization Plan for a Client Based on the Appliances That They Have, Habits They Have
  researcher = Agent(
        role="Appliance Energy Optimizer",
        goal="Your goal is to recommend AN ENERGY OPTIMIZATION PLAN FOR A CUSTOMER with habit changes, appliance setting changes, appliance brand recommendations, that can help her acheieve lower energy bills FROM THE GIVEN DATA ONLY based on a given customer's appliances and HABITS %s. DO NOT USE THE WEB ."""%(taste),
        backstory="""You are a Customer Preference Analyzer THAT specializes on OPTIMIZING ENERGY CONSUMPTION.""",
        verbose=True,
        allow_delegation=False,

    )


  #Agent For RECOMMEDNING Brands of Appliances That Are Energy Efficient
  brandexpert = Agent(
        role="Product Finder",
        goal="""ADD the Previous Agents Plan with Recommendations of Energy Efficient Products for each of the appliances listed from the last agent""",
        backstory="""You are a Product Finder Expert in the Energy Efficiency & Sustainability Domain, known for having ingenous skills in finding the right product for a client""",
        verbose=True,

        allow_delegation=False,
         )

  
   #Agent For Scraping Some General Energy SAVING hacks from the Website using Scraper Tool
  webscraper = Agent(
      role="Energy SAVING Hack recommender",
      goal="""Give some hacks for energy hacking from the scraped website""",
      backstory="""You are a Writer Known For writing compelling energy optimization plans to clients""",
      verbose=True,
      tool = [tool],

      allow_delegation=False)



  #Task For Creating A Personalized Energy Optimization Plan for the Customer
  task1 = Task(
      description="""recommend AN ENERGY OPTIMIZATION PLAN FOR A CUSTOMER with habit changes, appliance setting changes, appliance brand recommendations, that can help her acheieve lower energy bills FROM THE GIVEN DATA ONLY based on a given customer's appliances and HABITS %s. DO NOT USE THE WEB ."""%(taste),
      expected_output="A PLAN IN TEXT FORMAT",
      agent=researcher,
  )

  #Task For Recommending Energy Efficient Brands of Appliances
  task2 = Task(
      description="""ADD TO the Previous Agents Plan with Recommendations of Energy Efficient Products for each of the appliances listed from the last agent""",
      expected_output="PREVIOUS PLAN INCLUDING Brand Recommendations For Each of the Appliances.",
      agent=brandexpert
  )

  #Task For Extracting General Energy Saving Hacks From a Website
  task3 = Task(
      description="""ADD TO THE PREVIOUS AGENTS PLAN WITH Hacks for saving energy FOR the CUSTOMER based on the scraped data""",
      expected_output="PREVIOUS PLAN INCLUDING energy saving hacks from the scraped data",
      agent=webscraper,
  )

  
  #Define the crew based on the defined agents and tasks
  crew = Crew(
      agents=[researcher,brandexpert,webscraper],
      tasks=[task1,task2,task3],
      verbose=True,  # You can set it to 1 or 2 to different logging levels
  )

  result = crew.kickoff()
  return result



text = create_crew(textt)
print(text)
