!pip install llama-index-core
!pip install llama-index-readers-file
!pip install llama-index-embeddings-openai
!pip install llama-index-llms-llama-api
!pip install 'crewai[tools]'
!pip install llama-index-llms-langchain

import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from langchain_openai import ChatOpenAI

#Save Open AI Key as an environment variable
openai_api_key = ''
os.environ['OPENAI_API_KEY']=openai_api_key

#Load Data using LlamaIndex’s SimpleDirectoryReader
reader = SimpleDirectoryReader(input_files=["starbucks.csv"])
docs = reader.load_data()

# Create Query Tool For Interacting With the CSV Data
#we have used gpt-4o model here as the LLM. Other OpenAI models can also be used
llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1000)

#creates a VectorStoreIndex from a list of documents (docs)
index = VectorStoreIndex.from_documents(docs)

#The vector store is transformed into a query engine. 
#Setting similarity_top_k=5 limits the results to the top 5 documents that are most similar to the query, 
#llm specifies that the LLM should be used to process and refine the query results
query_engine = index.as_query_engine(similarity_top_k=5, llm=llm)
query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Coffee Promo Campaign",
    description="Use this tool to lookup the Starbucks Coffee Dataset",
)

#Creating the Crew Consisting of Multiple Agents 
def create_crew(taste):
  
  #Agent For Choosing 3 Types of Coffee Based on Customer Preferences
  researcher = Agent(
        role="Coffee Chooser",
        goal="Choose Starbucks Coffee based on customer preferences",
        backstory="""You work at Starbucks.
      Your goal is to recommend 3 Types of Coffee FROM THE GIVEN DATA ONLY based on a given customer's lifestyle tastes %s. DO NOT USE THE WEB to recommend the coffee types."""%(taste),
        verbose=True,
        allow_delegation=False,
        tools=[query_tool],
    )
    
  
  #Agent For Drafting Promotional Campaign based on Chosen Coffee
  writer = Agent(
        role="Product Content Specialist",
        goal="""Craft a Promotional Campaign that can mailed to customer based on the 3 Types of the Coffee suggested by the previous agent.Also GIVE ACCURATE  Starbucks Location in the given location in the query %s using 'web_search_tool' from the WEB where the customer can enjoy these coffees in the writeup"""%(taste),
        backstory="""You are a renowned Content Specialist, known for writing to customers for promotional campaigns""",
        verbose=True,

        allow_delegation=False)
  
  #Task For Choosing 3 Types of Coffee Based on Customer Preferences
  task1 = Task(
      description="""Recommend 3 Types of Coffee FROM THE GIVEN DATA ONLY based on a given customer's lifestyle tastes %s. DO NOT USE THE WEB to recommend the coffee types."""%(taste),
      expected_output="List of 3 Types of Coffee",
      agent=researcher,
  )
  
  #Task For Drafting Promotional Campaign based on Chosen Coffee
  task2 = Task(
      description="""Using ONLY the insights provided, develop a Promotional Campaign that can mailed to customer based on 3 Types of the Coffee suggested by the previous agent.

    Also GIVE ACCURATE Starbucks Location in the given location in the query %s using 'web_search_tool' from the WEB where the customer can enjoy these coffees in the writeup. Your writing should be accurate and to the point. Make it respectful and customer friendly"""%(taste),
      expected_output="Full Response to customer on how to resolve the issue .",
      agent=writer
  )
  
  #Define the crew based on the defined agents and tasks
  crew = Crew(
      agents=[researcher,writer],
      tasks=[task1,task2],
      verbose=True,  # You can set it to 1 or 2 to different logging levels
  )

  result = crew.kickoff()
  return result

#Checking Output For a Sample Customer
text = create_crew("Emily is from Gurgaon, India and likes High calorie coffees and prefers Latte more than Cappuccino.")
print(text)
