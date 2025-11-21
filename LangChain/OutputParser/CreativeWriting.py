from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field


import os;

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.5
)

writeup_instruction = """
Write a passage about our new product under 200 words.
Product: Jumbo EV Car
Title must have phrase "NextGen EV Car"
First sentence must have few among these words: innovative, cutting-edge
Passage must include features: long-range battery, fast charging, advanced safety
Last sentence should have reference to website "jumboev.com"
"""

template_1 = """
You are an expert in writing creative marketing material. 
Generate list of three plans for executing tasks with below instruction.
Each sentence should start with new line.
Minimum 4, Maximum 10 sentences expected

Instructions: {task_instructions}
Format_Instructions: {format_instructions}
"""

class PassagePlans(BaseModel):
    plan_Number: int = Field(description = "Plan Number")
    plan_steps: list[str] = Field(description = "Plan Steps" )

class PassagePlanList(BaseModel):
    plans: list[PassagePlans] = Field(description = "List of Passage Plans")

generate_plan_opParser = PydanticOutputParser(pydantic_object=PassagePlanList)

prompt_template_1 = PromptTemplate(
    template=template_1,
    input_variables = ["task_instructions"],
    partial_variables= {"format_instructions" : 
                        generate_plan_opParser.get_format_instructions()}
)

llm_chain_1 = RunnableSequence(prompt_template_1, llm, generate_plan_opParser)
response_1 = llm_chain_1.invoke({"task_instructions": writeup_instruction})

print(f"Total plans: {len(response_1.plans)}\n")
    
for plan in response_1.plans:
    print(f"{'='*60}")
    print(f"📋 PLAN {plan.plan_Number}")
    print(f"{'='*60}")
    for i, step in enumerate(plan.plan_steps, 1):
        print(f"{i}. {step}")



# Chain 2: Vote on best passage
template_2 = """
You are an expert marketing strategist evaluating different approaches.
Review the following plans and select the best one for creating compelling marketing content.

Original Task Instructions:
{task_instructions}

Plans to Evaluate:
{plans}

Carefully analyze each plan and select the best one.
Provide your selection with detailed reasoning.

Format_Instructions: {format_instructions}
"""

class BestPlanSelected(BaseModel):
     selected_best_plan: int = Field(description="Selected Best Plan Number")
     plan_steps: list[str] = Field(description = "Selected Best Plan's Steps" )
     reason: str = Field(description="Reason for the plan selection")

vote_plan_opParser = PydanticOutputParser(pydantic_object=BestPlanSelected)

prompt_template_2=PromptTemplate(template=template_2,
    input_variables=["task_instructions", "plans"],
    partial_variables={"format_instructions" : vote_plan_opParser.get_format_instructions()}
)
'''
plans_formatted = "\n\n".join([
    f"Plan {plan.plan_Number}:\n" + "\n".join([f"  - {step}" for step in plan.plan_steps])
    for plan in response_1.plans
])
'''
llm_chain_2 = RunnableSequence(prompt_template_2,llm, vote_plan_opParser)
response_2 = llm_chain_2.invoke({
    "task_instructions" : writeup_instruction,
    "plans" : response_1
})


print(f"\n{'='*60}")
print(f"Selected Plan: {response_2.selected_best_plan}")
print(f"Steps: ")
for i, step in enumerate(response_2.plan_steps, 1):
    print(f"{i}. {step}")
print(f"Reason: {response_2.reason}")
print(f"{'='*60}\n")


# 3. Generate 3 passages 
template_3 = """
You are an expert marketing copywriter.
Create THREE different passages following the selected plan and original instructions.

Original Task Instructions:
{task_instructions}

Selected Plan to Follow:
{selected_plan}

Generate three distinct, creative variations of the marketing passage.
Each passage should follow the plan and meet all the original requirements.

Format_Instructions: {format_instructions}
"""

class MarketingPlan(BaseModel):
    passage_number: int = Field(description = "Passage Number")
    title: str = Field(description="Title of Passage")
    content: str = Field(description = "Passage Content" )

class MarketingPlanList(BaseModel):
    passages: list[MarketingPlan] = Field(description = "List of Marketing Passage")

generate_passage_opParser = PydanticOutputParser(pydantic_object=MarketingPlanList)

prompt_template_3 = PromptTemplate(template=template_3,
    input_variables=["task_instructions", "selected_plan"],
    partial_variables={"format_instructions" : 
                       generate_passage_opParser.get_format_instructions()}
)

llm_chain_3 = RunnableSequence(prompt_template_3,
    llm,
    generate_passage_opParser)

response_3=llm_chain_3.invoke({
    "task_instructions": writeup_instruction,
    "selected_plan": response_2
})

print("Generated Passages:\n")
for passage in response_3.passages:
    print(f"Passage {passage.passage_number}:")
    print(f"Title: {passage.title}")
    print(f"Content: {passage.content}")
    print(f"\n{'-'*60}\n")

# Select Best Passage
template_4 = """
You are an expert marketing critic with deep understanding of compelling copy.
Review the three passages below and select the best one.

Original Task Instructions:
{task_instructions}

Passages to Evaluate:
{passages}

Evaluate each passage based on:
- Adherence to requirements
- Creativity and engagement
- Clarity and persuasiveness
- Overall marketing impact

Select the best passage and provide detailed reasoning.

Format_Instructions: {format_instructions}
"""

class BestPassage(BaseModel):
    selected_passage_number: int = Field(description="Number of the selected passage")
    reason: str = Field(description="Detailed reasoning for why this passage is the best")
    strengths: list[str] = Field(description="Key strengths of the selected passage")
    title: str = Field(description="Title of the Selected Passage")
    content: str = Field(description = "Content of the Selected Passage" )

select_passage_opParser = PydanticOutputParser(pydantic_object=BestPassage)

prompt_template_4=PromptTemplate(template=template_4,
    input_variables=["task_instructions", "passages"],
    partial_variables={"format_instructions" : 
                       select_passage_opParser.get_format_instructions})

llm_chain_4 = RunnableSequence(prompt_template_4,
    llm,
    select_passage_opParser)

response_4 = llm_chain_4.invoke({
    "task_instructions" : writeup_instruction,
    "passages" : response_3
})



print(f"{'='*60}")
print(f"FINAL SELECTION")
print(f"{'='*60}")
print(f"Selected Passage: {response_4.selected_passage_number}")
print(f"\nReason: {response_4.reason}")
print(f"\nTitle: {response_4.title}")
print(f"\nContent: {response_4.content}")
print(f"\nStrengths:")
for strength in response_4.strengths:
    print(f"  - {strength}")