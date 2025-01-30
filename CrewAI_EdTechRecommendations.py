#INSTALL & IMPORT NECESSARY LIBRARIES
!pip install crewai
!pip install langchain_openai


import os
import re
import json
import pandas as pd
from tqdm import tqdm
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from textwrap import dedent

#DEFINE LLM MODEL WITH API KEY
import os
openai_api_key = ''
os.environ['OPENAI_API_KEY'] = openai_api_key
llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0.1,
    max_tokens=8000
)


#DEFINE DATASETS TO BE USED
csv = '''Academic Goals, Major, Hobbies, Computer Skills, Interest in Languages, GPA
To become a software engineer, Computer Science, Gaming, Advanced, Spanish, 3.7
To study environmental science, Environmental Science, Hiking, Intermediate, French, 3.5
To pursue a career in medicine, Pre-Med, Playing the piano, Advanced, Spanish, 3.9
To major in psychology, Psychology, Reading, Intermediate, German, 3.6
To work in international relations, Political Science, Traveling, Basic, Mandarin, 3.8
To become a teacher, Education, Painting, Advanced, Spanish, 3.4
To study literature, English Literature, Writing, Intermediate, French, 3.9
To pursue a career in business, Business Administration, Playing soccer, Basic, Mandarin, 3.5
To become a biologist, Biology, Photography, Advanced, German, 3.7
To work in data analysis, Statistics, Cooking, Intermediate, Japanese, 3.6
'''

products = '''
"Introduction to Computer Science" - Offered by Harvard University on edX
"Biology: Life on Earth" - Offered by Coursera
"Introduction to Psychology" - Offered by Yale University on Coursera
"Environmental Science" - Offered by University of Leeds on FutureLearn
"Introduction to Literature" - Offered by MIT on edX
"Medical Terminology" - Offered by University of Pittsburgh on Coursera
"Data Science and Machine Learning" - Offered by Stanford University on Coursera
"Cell Biology" - Offered by Massachusetts Institute of Technology on edX
"Positive Psychology" - Offered by University of North Carolina at Chapel Hill on Coursera
"Environmental Law and Policy" - Offered by Vermont Law School on Coursera
"Programming for Everybody (Getting Started with Python)" - Offered by University of Michigan on Coursera
"Anatomy: Human Neuroanatomy" - Offered by University of Michigan on Coursera
"Introduction to Cognitive Psychology" - Offered by Duke University on Coursera
"Climate Change and Health: From Science to Action" - Offered by Harvard University on edX
"English for Science, Technology, Engineering, and Mathematics" - Offered by University of Pennsylvania on Coursera
"An Introduction to American Law" - Offered by University of Pennsylvania on Coursera
"Introduction to Chemistry: Reactions and Ratios" - Offered by Duke University on Coursera
"Epidemiology: The Basic Science of Public Health" - Offered by University of North Carolina at Chapel Hill on Coursera
"Computer Science: Programming with a Purpose" - Offered by Princeton University on Coursera
"Introduction to Statistics and Data Analysis" - Offered by Rice University on Coursera
"Genes and the Human Condition (From Behavior to Biotechnology)" - Offered by University of Maryland on Coursera
"Ethics, Technology, and the Future of Medicine" - Offered by Georgetown University on edX
"Fundamentals of Immunology" - Offered by Harvard University
'''

# DEFINE AGENTS TO BE USED
student_profiler = Agent(
  role='student_profiler',
  goal='''From limited data, you logically deduct conclusions about students.''',
  backstory='You are an expert psychologist with decades of experience.',
  llm = llm,allow_delegation=False,verbose=True)

course_specialist = Agent(
     role='course specialist',
     goal='''Match the suitable course to the students''',
     backstory='You have exceptional knowledge of the courses and can say how valuable they are to a student.',
     llm = llm,allow_delegation=False,verbose=True)

Chief_Recommendation_Director = Agent(
     role="Chief Recomeendation Director",
     goal=dedent("""\Oversee the work done by your team to make sure it's the best
		possible and aligned with the course's goals, review, approve,
		ask clarifying question or delegate follow up work if necessary to make
		decisions"""),
     backstory=dedent("""\You're the Chief Promotion Officer of a large EDtech company. You're launching a personalized ad campaign,
          trying to make sure your team is crafting the best possible
	  content for the customer."""),
     llm = llm,tools=[],allow_delegation=False, verbose=True)

campaign_agent = Agent(
     role="campaign_agent",
     goal=dedent("""\Develop compelling and innovative content
		for ad campaigns, with a focus customer specific ad copies."""),
     backstory=dedent("""\As a Creative Content Creator at a top-tier
			digital marketing agency, you excel in crafting advertisements
			that resonate with potential customers.
			Your expertise lies in turning marketing strategies
			into engaging stories that capture
			attention and inspire buying action."""),
     llm = llm,allow_delegation=False, verbose=True)

#DEFINE TASKS FOR EACH AGENTS
def get_ad_campaign_task(agent, customer_description, products):
  return Task(description=dedent(f"""\
    You're creating a targeted marketing campaign tailored to what we know about our student customers.

    For each student customer, we have to choose exactly three courses to promote in the next campaign.
    Make sure the selection is the best possible and aligned with the student customer,
	  review, approve, ask clarifying question or delegate follow up work if
		necessary to make decisions. When delegating work send the full draft
		as part of the information.
    This is the list of all the courses participating in the campaign: {products}.
    This is all we know so far from the student customer: {customer_description}.

    To start this campaign we will need to build first an understanding of our student customer.
    Once we have a profile about the student customers interests, lifestyle and means and needs,
    we have to select exactly three courses that have the highest chance to be bought by them.

    Your final answer MUST be exactly 3 courses from the list, each with a short description
    why it matches with this student customer. It must be formatted like this example:
     :
     :
     :
    """),
    agent=agent,expected_output='A refined finalized version of the marketing campaign in markdown format'
  )

def get_ad_campaign_written_task(agent, selection):
    return Task(description=dedent(f"""\
    You're creating a targeted marketing campaign tailored to what we know about our student customer.

    For each student customer, we have chosen three courses to promote in the next campaign.
    This selection is tailored specifically to the customer: {selection},

    To end this campaign succesfully we will need a promotional message advertising these courses  to the student customer with the ultimate intent that they buy from us.
    This message should be around 3 paragraphs, so that it can be easily integrated into the full letter. For example:
    Interested in learning data science, get yourself enrolled in this course from Harvard University.
    Take Your career to the next level with the help of this course.

    You need to review, approve, and delegate follow up work if necessary to have the complete promotional message. When delegating work send the full draft
		as part of the information.

    Your final answer MUST include the 3 products from the list, each with a short promotional message.
    """),
    agent=agent,expected_output='A refined finalized version of the marketing campaign in markdown format'
  )


#ITERATE THROUGH EACH ROW OF STUDENT PROFILE DATAFRAME
df_output_list = [] #to store results

for index, row in df_customers.iterrows():
  print('############################################## '+ str(index))
  customer_description = f'''
  Their academic goals are {row['Academic Goals']}.
  Their major is in {row[' Major']}.
  Their Hobbies are {row[' Hobbies']}.
  Their computer skills are {row[' Computer Skills']}.
  Their interest in languages are {row[' Interest in Languages']}.
  Their GPA is {row[' GPA']}.
  '''
  print(customer_description)
  
  #Define Task 1 for selecting top 3 relevant products
  
  task1 = get_ad_campaign_task(Chief_Promotional_Director,customer_description, products)
  #start crew
  targetting_crew = Crew(
    agents=[student_profiler, product_specialist,Chief_Promotional_Director],
    tasks=[task1],
    verbose=True, 
  process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
  )
  targetting_result = targetting_crew.kickoff()
  
  #Define Task 2 for Generating Recommendation Campaign
  
  task2 = get_ad_campaign_written_task(Chief_Promotional_Director,targetting_result)
  copywriting_crew = Crew(
    agents=[campaign_agent,Chief_Promotional_Director],
    tasks=[task2],
    verbose=True, 
  process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
  )
  copywriting_result = copywriting_crew.kickoff()
  #create one line in output df
  df_output_list.append({'customer':customer_description,
                         'targeted_products':targetting_result,
                         'promo_msg':copywriting_result,
                        })

#loop ended, collect results in dataframe
df_output = pd.DataFrame(df_output_list)
