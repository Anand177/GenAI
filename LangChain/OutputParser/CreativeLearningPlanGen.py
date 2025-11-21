from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

task_instructions = """
Write a passage about learning plan under 200 words
Learning plan: Graduate in Computer Science, aspiring to be a data scientist in two years
The person is proficient in computer science& engineering basics
Add latest programming languages
Must refer at least one cloud computing platform
Plan with Machine learning/ Generative AI would be preferred
"""

## 1. Generate 3 Learning Plans
plan_list_template = """
You are an expert in generating learning path for college graduates. 
Generate list of three plans for executing tasks with below instruction.
Each sentence should start with new line.
Minimum 5, Maximum 10 sentences expected

Instructions: {task_instructions}
Format_Instructions: {format_instructions}
"""

# Classes for learning plans
class LearningPlans(BaseModel):
    plan_number: int = Field(description = "Plan Number")
    plan_steps: list[str] = Field(description = "Plan Steps" )

class LearningPlanList(BaseModel):
    plans: list[LearningPlans] = Field(description = "List of Learning Plans")

plan_list_op_parser = PydanticOutputParser(pydantic_object=LearningPlanList)

plan_prompt_template = PromptTemplate(template=plan_list_template,
    input_variables=["task_instructions"],
    partial_variables={"format_instructions": plan_list_op_parser.get_format_instructions}
)

plan_list_chain = RunnableSequence(plan_prompt_template, llm, plan_list_op_parser)
plan_list_response = plan_list_chain.invoke({"task_instructions" : task_instructions})

print("Learning Plans generated. Printing...")
for plan in plan_list_response.plans:
    print(f"{'='*60}")
    print(f"📋 PLAN {plan.plan_number}")
    print(f"{'='*60}")
    for i, step in enumerate(plan.plan_steps, 1):
        print(f"{i}. {step}")

## 1. Learning Plan Generation Complete


## 2. Vote on the Best Learning Plans
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
    plan_number: int = Field(description = "Selected Plan's Number")
    plan_steps: list[str] = Field(description = "Selected Plan's Steps" )
    reason: str = Field(description="Reason for the plan selection")

voted_plan_op_parser = PydanticOutputParser(pydantic_object=BestLearningPlan)

voted_plan_prompt_template = PromptTemplate(template=plan_vote_template,
    input_variables=["task_instructions", "plans"],
    partial_variables={"format_instructions": voted_plan_op_parser.get_format_instructions}
)

voted_plan_chain = RunnableSequence(voted_plan_prompt_template, llm, voted_plan_op_parser)
voted_plan_response = voted_plan_chain.invoke({"task_instructions" : task_instructions,
                        "plans" : plan_list_response})

print(f"\n{'='*60}")
print(f"Selected Plan: {voted_plan_response.plan_number}")
print(f"Steps: ")
for i, step in enumerate(voted_plan_response.plan_steps, 1):
    print(f"{i}. {step}")
print(f"Reason: {voted_plan_response.reason}")
print(f"{'='*60}\n")

## 2. Vote on the Best Learning Plans Complete

## 3. Generate 3 Learning Path for best learning plan
path_list_template = """
You are an expert learning instructor.
Create THREE different learning paths following the selected plan and original instructions.
Include online courses, classes, open source material links to complete the learning plan
Appropriate tools should be included
Steps with certification, hackathon, competition recommended
Add reference to online and opensource tools and platforms for learning

Original Task Instructions:
{task_instructions}

Selected Plan to Follow:
{selected_plan}

Generate three distinct, creative variations of the learning plan.
Each learning path should follow the plan and meet all the original requirements.

Format_Instructions: {format_instructions}
"""

# Classes for learning paths
class LearningPaths(BaseModel):
    path_number: int = Field(description = "Path Number")
    path_steps: list[str] = Field(description = "Path Steps" )

class LearningPathList(BaseModel):
    pathList: list[LearningPaths] = Field(description = "List of Learning Paths")

path_list_op_parser = PydanticOutputParser(pydantic_object=LearningPathList)

path_prompt_template = PromptTemplate(template=path_list_template,
    input_variables=["task_instructions", "selected_plan"],
    partial_variables={"format_instructions": path_list_op_parser.get_format_instructions}
)

path_list_chain = RunnableSequence(path_prompt_template, llm, path_list_op_parser)
path_list_response = path_list_chain.invoke({"task_instructions" : task_instructions,
            "selected_plan" : voted_plan_response})

print("Learning Paths generated. Printing...")
for path in path_list_response.pathList:
    print(f"{'='*60}")
    print(f"📋 PATH {path.path_number}")
    print(f"{'='*60}")
    for i, step in enumerate(path.path_steps, 1):
        print(f"{i}. {step}")

## 3. Completion of Learning Path generation


## 4. Vote on the Best Learning Path
path_vote_template = """
You are an expert education consultant evaluating different learning path.
Review the following plans and select the best one for complelling Data scientist career.

Original Task Instructions:
{task_instructions}

Plans to Evaluate:
{plans}
- Creativity and engagement
- Clarity and persuasiveness


Carefully analyze each plan and select the best one.
- Adherence to requirements
- Plan Practicality and feasibility 
- Plan Clarity and Engagement 
- Overall knowledge acquisition and impact

Provide your selection with detailed reasoning.

Format_Instructions: {format_instructions}
"""

# Classes for learning paths
class BestLearningPath(BaseModel):
    path_number: int = Field(description = "Selected Path's Number")
    path_steps: list[str] = Field(description = "Selected Path's Steps" )
    reason: str = Field(description="Reason for the path selection")

voted_path_op_parser = PydanticOutputParser(pydantic_object=BestLearningPath)

voted_path_prompt_template = PromptTemplate(template=path_vote_template,
    input_variables=["task_instructions", "plans"],
    partial_variables={"format_instructions": voted_path_op_parser.get_format_instructions}
)

voted_path_chain = RunnableSequence(voted_path_prompt_template, llm, voted_path_op_parser)
voted_path_response = voted_path_chain.invoke({"task_instructions" : task_instructions,
                        "plans" : path_list_response})

print(f"\n{'='*60}")
print(f"Selected Plan: {voted_path_response.path_number}")
print(f"Steps: ")
for i, step in enumerate(voted_path_response.path_steps, 1):
    print(f"{i}. {step}")
print(f"Reason: {voted_path_response.reason}")
print(f"{'='*60}\n")

## 4. Vote on the Best Learning Path Complete
