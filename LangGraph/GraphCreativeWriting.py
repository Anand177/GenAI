from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing import Annotated, List, TypedDict

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class LearningPlans(BaseModel):
    model_name: str = Field(description = "Name of LLM model that generated the plan")
    plan_steps: list[str] = Field(description = "Plan Steps" )

class LearningPlanList(BaseModel):
    plan_list: list[LearningPlans] = Field(description = "List of Generated Plans")

# Classes for learning plans
class BestLearningPlan(BaseModel):
    model_name: str = Field(description = "Name of LLM model that generated the plan")
    plan_steps: list[str] = Field(description = "Plan Steps" )
    reason: str = Field(description="Reason for the plan selection")

# Classes for learning paths
class LearningPaths(BaseModel):
    path_number: int = Field(description = "Path Number")
    path_steps: list[str] = Field(description = "Path Steps" )

class LearningPathList(BaseModel):
    pathList: list[LearningPaths] = Field(description = "List of Learning Paths")

class BestLearningPath(BaseModel):
    model_name: str = Field(description = "Name of LLM model that generated the path")
    path_steps: list[str] = Field(description = "Path Steps" )
    reason: str = Field(description="Reason for the path selection")


def append_plan_list(current: List[LearningPlans], new: List[LearningPlans]) -> List[LearningPlans]:
    return current + new

def append_path_list(current: List[LearningPaths], new: List[LearningPaths]) -> List[LearningPaths]:
    return current + new

class GenState(TypedDict):
    num_workers: int
    plan_list: Annotated[List[LearningPlans], append_plan_list]
    best_plan: BestLearningPlan
    path_list: Annotated[List[LearningPaths], append_path_list]
    best_path: BestLearningPath

class PlanGenState(TypedDict):
    worker_name: str
    llm_name: str
    plan_list: Annotated[List[LearningPlans], append_plan_list]      # Shared bet parent& workers

class PathGenState(TypedDict):
    worker_name: str
    llm_name: str
    best_plan: BestLearningPlan                                     # Shared bet parent& workers
    path_list: Annotated[List[LearningPaths], append_path_list]      # Shared bet parent& workers


groq_llm = [ ChatGroq(
    model="openai/gpt-oss-20b", # llama-3.1-70b-versatile
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
), ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)]

judge_llm = GoogleGenerativeAI(
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

## 1. Generate Learning Plans
plan_list_template = """
You are an expert in generating learning path for college graduates. 
Generate a plan for executing tasks with below instruction.
Each sentence should start with new line.
Minimum 5, Maximum 10 sentences expected

Instructions: {task_instructions}
Format_Instructions: {format_instructions}
"""

plan_list_op_parser = PydanticOutputParser(pydantic_object=LearningPlans)
plan_prompt_template = PromptTemplate(template=plan_list_template,
    input_variables=["task_instructions"],
    partial_variables={"format_instructions": plan_list_op_parser.get_format_instructions}
)

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

path_list_template = """
You are an expert learning instructor.
Create a learning path following the selected plan and original instructions.
Include online courses, classes, open source material links to complete the learning plan
Appropriate tools should be included
Steps with certification and competition recommended
Add reference to online and opensource tools and platforms for learning
Learning path should follow the plan and meet all the original requirements.

Original Task Instructions: {task_instructions}

Selected Plan to Follow: {selected_plan}

Format_Instructions: {format_instructions}
"""

voted_plan_op_parser = PydanticOutputParser(pydantic_object=BestLearningPlan)

voted_plan_prompt_template = PromptTemplate(template=plan_vote_template,
    input_variables=["task_instructions", "plans"],
    partial_variables={"format_instructions": voted_plan_op_parser.get_format_instructions}
)

path_list_op_parser = PydanticOutputParser(pydantic_object=LearningPaths)

path_prompt_template = PromptTemplate(template=path_list_template,
    input_variables=["task_instructions", "selected_plan"],
    partial_variables={"format_instructions": path_list_op_parser.get_format_instructions}
)

path_vote_template = """
You are an expert education consultant evaluating different learning path.
Review the following plans and select the best one for complelling Data scientist career.

Original Task Instructions: {task_instructions}

Paths to Evaluate: {paths}

Carefully analyze each plan and select the best one.
- Adherence to requirements
- Plan Practicality and feasibility 
- Plan Clarity and Engagement 
- Overall knowledge acquisition and impact

Provide your selection with detailed reasoning.

Format_Instructions: {format_instructions}
"""

voted_path_op_parser = PydanticOutputParser(pydantic_object=BestLearningPath)

voted_path_prompt_template = PromptTemplate(template=path_vote_template,
    input_variables=["task_instructions", "paths"],
    partial_variables={"format_instructions": voted_path_op_parser.get_format_instructions}
)


def orchestrator_node(state : GenState) -> GenState:
    return {"plan_list" : []}        # Init Parent State

def plan_gen_node(state: PlanGenState):
    
    print(f"Worker -> {state['worker_name']} using LLM : {state['llm_name']}")
    worker_idx = int(state['worker_name'].split('#')[1]) - 1
    current_llm = groq_llm[worker_idx]

    worker_chain = plan_prompt_template | current_llm | plan_list_op_parser
    resp =worker_chain.invoke({"task_instructions" : task_instructions})

    return {"plan_list" : [resp]}

# Router Logic 
def assign_plan_gen_workers(state: GenState) -> GenState:
    num_workers = state.get('num_workers', 2)          # Get # of workers to create
    workers = []

    for i in range(num_workers):
        worker_args = {
            "worker_name" : f"Worker#{i+1}",
            "llm_name" : groq_llm[i].model_name
        }
        workers.append(Send("plan_gen_node", worker_args))
    return workers

def path_gen_node(state: PathGenState):
    
    print(f"Worker -> {state['worker_name']} using LLM : {state['llm_name']}")
    worker_idx = int(state['worker_name'].split('#')[1]) - 1
    current_llm = groq_llm[worker_idx]

    worker_chain = path_prompt_template | current_llm | path_list_op_parser
    resp =worker_chain.invoke({"task_instructions" : task_instructions,
                "selected_plan": state['best_plan']})
    print(resp)

    return {"path_list" : [resp]}

def assign_path_gen_workers(state: GenState) -> GenState:
    num_workers = state.get('num_workers', 2)          # Get # of workers to create
    workers = []

    for i in range(num_workers):
        worker_args = {
            "worker_name" : f"Worker#{i+1}",
            "llm_name" : groq_llm[i].model_name,
            "best_plan" : state["best_plan"]
        }
        workers.append(Send("path_gen_node", worker_args))
    return workers

def plan_judge_node(state: GenState) -> GenState:
    
    # Printing Generated Learning Plans
    for plan in state["plan_list"]:
        print(f"{'='*60}")
        print(f"📋 PLAN {plan.model_name}")
        print(f"{'='*60}")
        for i, step in enumerate(plan.plan_steps, 1):
            print(f"{i}. {step}") 

    # Judgling the best plan 
    voted_plan_chain = voted_plan_prompt_template | judge_llm | voted_plan_op_parser
    voted_plan_response = voted_plan_chain.invoke({"task_instructions" : task_instructions,
            "plans" : state["plan_list"]})

    print(f"\n{'='*60}")
    print(f"Selected Plan: {voted_plan_response.model_name}")
    print(f"Steps: ")
    for i, step in enumerate(voted_plan_response.plan_steps, 1):
        print(f"{i}. {step}")
    print(f"Reason: {voted_plan_response.reason}")
    print(f"{'='*60}\n")

    return {"best_plan": voted_plan_response}

def path_judge_node(state: GenState) -> GenState:
    
    print(f"\n{'='*60}")
    print("⚖️  JUDGING THE BEST LEARNING PATH")
    print(f"{'='*60}")

    for path in state["path_list"]:
        print(f"{'='*60}")
        for i, step in enumerate(path.path_steps, 1):
            print(f"{i}. {step}") 
    
    voted_path_chain = voted_path_prompt_template | judge_llm | voted_path_op_parser
    voted_path_response = voted_path_chain.invoke({"task_instructions" : task_instructions,
            "paths" : state["path_list"]})

    print(f"\n{'='*60}")
    print(f"Selected Path: {voted_path_response.model_name}")
    print(f"Steps: ")
    for i, step in enumerate(voted_path_response.path_steps, 1):
        print(f"{i}. {step}")
    print(f"Reason: {voted_path_response.reason}")
    print(f"{'='*60}\n")

    return {"best_path": voted_path_response}


graph = StateGraph(GenState)
# Add Workers
graph.add_node("orchestrator_node", orchestrator_node)
graph.add_node("plan_gen_node", plan_gen_node)
graph.add_node("plan_judge_node", plan_judge_node)
graph.add_node("path_gen_node", path_gen_node)
graph.add_node("path_judge_node", path_judge_node)

# Add Edges
graph.add_edge(START, "orchestrator_node")
graph.add_conditional_edges("orchestrator_node", assign_plan_gen_workers, ["plan_gen_node"])
graph.add_edge("plan_gen_node", "plan_judge_node")
graph.add_conditional_edges("plan_judge_node", assign_path_gen_workers, ["path_gen_node"])
graph.add_edge("path_gen_node", "path_judge_node")
graph.add_edge("path_judge_node", END)

graph_compiled = graph.compile()
response=graph_compiled.invoke({"num_workers": 2})

print(response)