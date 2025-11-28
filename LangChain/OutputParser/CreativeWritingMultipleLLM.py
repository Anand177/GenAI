from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

from pydantic import BaseModel, Field

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

judge_llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

groq_llm = [ ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
), ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)]
#    print(groq_llm[0].model_name)
#    print(groq_llm[1].model_name)


task_instructions = """
Write a passage about learning plan under 200 words
Learning plan: Graduate in Computer Science, aspiring to be a data scientist in two years
The person is proficient in computer science& engineering basics
Add latest programming languages
Must refer at least one cloud computing platform
Plan with Machine learning/ Generative AI would be preferred
"""

## 1. Generate Learning Plans
plan_list_template = """
You are an expert in generating learning path for college graduates. 
Generate a plan for executing tasks with below instruction.
Each sentence should start with new line.
Minimum 5, Maximum 10 sentences expected

Instructions: {task_instructions}
Format_Instructions: {format_instructions}
"""

# Classes for learning plans
class LearningPlans(BaseModel):
    model_name: str = Field(description = "Name of LLM model that generated the plan")
    plan_steps: list[str] = Field(description = "Plan Steps" )

class LearningPlanList(BaseModel):
    plan_list: list[LearningPlans] = Field(description = "List of Generated Plans")

plan_list_op_parser = PydanticOutputParser(pydantic_object=LearningPlans)

plan_prompt_template = PromptTemplate(template=plan_list_template,
    input_variables=["task_instructions"],
    partial_variables={"format_instructions": plan_list_op_parser.get_format_instructions}
)

plan_list_chain = [ RunnableSequence(plan_prompt_template, groq_llm[0], plan_list_op_parser),
            RunnableSequence(plan_prompt_template, groq_llm[1], plan_list_op_parser) ]

learning_plan_list = LearningPlanList(plan_list=[])
for chain in plan_list_chain:
    learning_plan_list.plan_list.append(chain.invoke({"task_instructions" : task_instructions}))


print("Learning Plans generated. Printing...")
for plan in learning_plan_list.plan_list:
    print(f"{'='*60}")
    print(f"📋 PLAN {plan.model_name}")
    print(f"{'='*60}")
    for i, step in enumerate(plan.plan_steps, 1):
        print(f"{i}. {step}")




plan_vote_template = """
You are an expert education strategist evaluating different learning plans.
Review the following plans and select the best one for complelling Data scientist career.

Original Task Instructions:
{task_instructions}

Plans to Evaluate:
{plans}

Carefully analyze each plan and select the best one.
Provide your selection with detailed reasoning.

Format_Instructions: {format_instructions}
"""

# Classes for learning plans
class BestLearningPlan(BaseModel):
    model_name: str = Field(description = "Name of LLM model that generated the plan")
    plan_steps: list[str] = Field(description = "Plan Steps" )
    reason: str = Field(description="Reason for the plan selection")

voted_plan_op_parser = PydanticOutputParser(pydantic_object=BestLearningPlan)

voted_plan_prompt_template = PromptTemplate(template=plan_vote_template,
    input_variables=["task_instructions", "plans"],
    partial_variables={"format_instructions": voted_plan_op_parser.get_format_instructions}
)

voted_plan_chain = RunnableSequence(voted_plan_prompt_template, judge_llm, voted_plan_op_parser)
voted_plan_response = voted_plan_chain.invoke({"task_instructions" : task_instructions,
                        "plans" : learning_plan_list})

print(f"\n{'='*60}")
print(f"Selected Plan: {voted_plan_response.model_name}")
print(f"Steps: ")
for i, step in enumerate(voted_plan_response.plan_steps, 1):
    print(f"{i}. {step}")
print(f"Reason: {voted_plan_response.reason}")
print(f"{'='*60}\n")
