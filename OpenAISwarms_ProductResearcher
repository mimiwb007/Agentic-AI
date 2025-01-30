# INSTALL THE LIBRARY
!pip install git+https://github.com/openai/swarm.git 
!pip install wikipedia

# DEFINE THE OPENAI API KEY
import os
os.environ['OPENAI_API_KEY']=''

# IMPORT NECESSARY PYTHON LIBRARIES
from swarm import Swarm, Agent
import wikipedia

# DEFINING THE FUNCTIONS
def transfer_to_product_agent():
    return product_agent 

def wikipedia_lookup(context_variables):
    try: 
      print("context variable",context_variables["report_text"])
      summ = wikipedia.page(context_variables["report_text"]).summary
      print("SUMM",summ)
      return summ
    except: return None

# DEFINING THE AGENTS
product_agent = Agent(
    name="Product Agent",
    instructions="LIST NAME OF the key PRODUCTS BY THE BRAND ONLY FROM THE RETRIEVED WIKIPEDIA INFORMATION in Bullet points. ONLY USE THE RETRIEVED INFORMATION to list the key products.DO NOT USE INFORMATION FROM OUTSIDE",
)


wiki_agent = Agent(
    name="Agent",
    instructions="""You are a helpful agent that answers user queries by finding and analysing information from Wikipedia.
                    You will be given a BRAND NAME and you must retrieve it's entry on Wikipedia and then hand over to the Summary Agent.""",
    functions=[wikipedia_lookup, transfer_to_product_agent],
)

#EXECUTE THE MULTI-AGENT SYSTEM
client = Swarm()
# Run summary agent
text = "philips"
response = client.run(
    agent=wiki_agent ,
    messages=[{"role": "user", "content": text}],
    context_variables={"report_text": text}
)
print(response.messages[-1]["content"])
