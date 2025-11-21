from dotenv import load_dotenv
from operator import itemgetter

from IPython.display import JSON
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

import os


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

writeup_instructions = """
Write a passage about our new product under 200 words.
Product: Anand EV Car
Title must have phrase "NextGen EV Car"
First sentence must have few among these words: innovative, cutting-edge
Passage must include features: long-range battery, fast charging, advanced safety
Last sentence should have reference to website "anandev.com"
"""


google_flash_llm_chain=GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.5
)
'''
groq_llama_llm_chain= ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.9
)

groq_gpt_llm_chain= ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-20b",
    temperature=0.9
)
'''
ouput_passage_indicator = "Passage:"
ouput_plan_indicator = "Plan:"

template_1 = """
Generate three separate step by step plans for executing the given task with the following task instructions.

Task Instructions:
{task_instructions}

{ouput_plan_indicator}
Your plan here.

""" 

prompt_template_1 = PromptTemplate(
    template = template_1,
    input_variables = ["task_instructions"],
    partial_variables = {"ouput_plan_indicator":ouput_plan_indicator}
)

chain_1_seq = prompt_template_1 | google_flash_llm_chain

generate_plan_chain_1 = RunnableParallel(
    task_instructions = itemgetter("task_instructions"),
    proposed_plans = chain_1_seq
)

#response = generate_plan_chain_1.invoke({"task_instructions": writeup_instructions})
#print("Generated Plans:\n", JSON(response).data)

output_best_plan_indicator = "Selected Plan:"

template_2 = """
Three  expert marketing writers were asked to put together a plan for the following task. 
Your job is to pick the best plan.

Task Instructions:
{task_instructions}

Proposed Plans:
{proposed_plans}

Your out should be in following format.

{output_best_plan_indicator} 
Put the plan steps here.

"""


prompt_template_2 = PromptTemplate(
    template = template_2,
    input_variables = ["task_instructions", "proposed_plans"],
    partial_variables = {"output_best_plan_indicator":output_best_plan_indicator}
)
#print(prompt_template_2.format(
#    task_instructions=writeup_instructions,proposed_plans="Dummy Plans"))

# Setup a sequential chain with prompt & LLM
chain_2_sequence = prompt_template_2 |  google_flash_llm_chain

# We will use the RunnableParallel so we can pass on the task_instructions & proposed plans as is to the output
vote_on_plan_chain_2 = RunnableParallel(
    task_instructions=itemgetter('task_instructions'), 
    proposed_plans = itemgetter('proposed_plans'),
    best_voted_plan = chain_2_sequence
)

chain_1_chain_2 =  generate_plan_chain_1  | vote_on_plan_chain_2

response = chain_1_chain_2.invoke({"task_instructions":writeup_instructions})

#print("Best Voted Plan:\n", JSON(response).data)

template_3 = """
Create three passages using the following task instructions and the plan.

Task Instructions:
{task_instructions}

Plan:
{best_voted_plan}

Your output will be in following format.

{ouput_passage_indicator}
Put the passage here.

"""

prompt_template_3 = PromptTemplate(
    template = template_3,
    input_variables = ["best_voted_plan"],
    partial_variables = {"ouput_passage_indicator" : ouput_passage_indicator, "task_instructions": writeup_instructions},
)

#print(prompt_template_3.format(best_voted_plan="DUMMY PLAN"))

# Setup a sequential chain with prompt & LLM
chain_3_sequence = prompt_template_3 |  google_flash_llm_chain

# We will use the RunnableParallel so we can pass on the task_instructions & proposed plans as is to the output
create_3_passages_chain_3 = RunnableParallel(
    task_instructions=itemgetter('task_instructions'), 
    proposed_plans = itemgetter('proposed_plans'),
    best_voted_plan = itemgetter('best_voted_plan'),
    passages = chain_3_sequence
)

chain_1_chain_2_chain_3 =  generate_plan_chain_1  | vote_on_plan_chain_2 | create_3_passages_chain_3
response = chain_1_chain_2_chain_3.invoke({"task_instructions":writeup_instructions})

#print("Generated Passages:\n", JSON(response).data)

output_best_passage_indicator = "Selected Passage:"

template_4 = """
Three experts were given the following task.

Task:
{task_instructions}

You need to review the 3 passages and identify ONE best passage out of the three. 

Proposed Passages:
{passages}

Your output will be in following format.

{output_best_passage_indicator}
Put the passage here.

"""

prompt_template_4 = PromptTemplate(
    template = template_4,
    input_variables = ["passages"],
    partial_variables = {"task_instructions": writeup_instructions, "output_best_passage_indicator" : output_best_passage_indicator, },
)

# print(prompt_template_4.format(passages = "DUMMY PASSAGES"))

# Setup a sequential chain with prompt & LLM
chain_4_sequence = prompt_template_4 |  google_flash_llm_chain

# We will use the RunnableParallel so we can pass on the task_instructions & proposed plans as is to the output
create_creative_passage_chain_4 = RunnableParallel(
    task_instructions=itemgetter('task_instructions'), 
    proposed_plans = itemgetter('proposed_plans'),
    best_voted_plan = itemgetter('best_voted_plan'),
    passages = itemgetter('passages'),
    creative_passage = chain_4_sequence
)

creative_writing_chain = generate_plan_chain_1 | vote_on_plan_chain_2 | create_3_passages_chain_3 | create_creative_passage_chain_4

response = creative_writing_chain.invoke({"task_instructions":writeup_instructions})
print("Creative Writing - Best Passage:\n", JSON(response).data)