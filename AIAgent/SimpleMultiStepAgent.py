from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_tavily.tavily_search import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import json

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize tools
tavily_tool = TavilySearch(
    include_raw_content=True, 
    search_depth="basic", 
    include_images=False, 
    max_results=3,
    include_answer=False
)
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = {
    "tavily_search": tavily_tool,
    "wikipedia": wikipedia_tool
}
                                   
# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GOOGLE_API_KEY, 
    temperature=0.1
)

# Planning prompt
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a planning assistant. Break down the user's question into a sequence of steps.
Each step should specify:
1. The action to take (search_tavily or search_wikipedia)
2. The query to use
3. Why this step is needed

Return your plan as a JSON array of steps. Each step should have: {{"step_number": int, "action": str, "query": str, "reason": str}}

Example:
[
  {{"step_number": 1, "action": "search_tavily", "query": "Halley's comet last sighting year", "reason": "Find when Halley's comet was last visible"}},
  {{"step_number": 2, "action": "search_wikipedia", "query": "Tamil Nadu Chief Minister 1986", "reason": "Find who was CM during that year"}}
]

Respond ONLY with the JSON array, no other text."""),
    ("human", "{input}")
])

# Execution prompt
execution_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Based on the execution results from all steps, provide a comprehensive answer to the original question.

Execution History:
{execution_history}

Original Question: {original_question}

Synthesize the information and provide a clear, concise answer."""),
    ("human", "Please provide the final answer.")
])

def create_plan(query: str) -> List[Dict]:
    """Generate a plan for answering the query"""
    chain = planning_prompt | llm | StrOutputParser()
    plan_text = chain.invoke({"input": query})
    
    # Clean up potential markdown formatting
    plan_text = plan_text.strip()
    if plan_text.startswith("```json"):
        plan_text = plan_text[7:]
    if plan_text.startswith("```"):
        plan_text = plan_text[3:]
    if plan_text.endswith("```"):
        plan_text = plan_text[:-3]
    plan_text = plan_text.strip()
    
    try:
        plan = json.loads(plan_text)
        return plan
    except json.JSONDecodeError as e:
        print(f"Error parsing plan: {e}")
        print(f"Raw plan text: {plan_text}")
        # Fallback plan
        return [
            {"step_number": 1, "action": "search_tavily", "query": query, "reason": "Search for information"}
        ]

def execute_step(step: Dict) -> str:
    """Execute a single step of the plan"""
    action = step["action"]
    query = step["query"]
    
    print(f"\n{'='*60}")
    print(f"Executing Step {step['step_number']}")
    print(f"Action: {action}")
    print(f"Query: {query}")
    print(f"Reason: {step['reason']}")
    print(f"{'='*60}")
    
    try:
        if action == "search_tavily":
            result = tools["tavily_search"].invoke({"query": query})
        elif action == "search_wikipedia":
            result = tools["wikipedia"].invoke(query)
        else:
            result = f"Unknown action: {action}"
        
        return str(result)
    except Exception as e:
        return f"Error executing {action}: {str(e)}"

def run_multistep_agent(query: str):
    """Run the multi-step plan-and-execute agent"""
    
    print(f"\n{'#'*60}")
    print(f"QUERY: {query}")
    print(f"{'#'*60}\n")
    
    # Step 1: Create Plan
    print("📋 CREATING PLAN...")
    plan = create_plan(query)
    
    print(f"\n📝 Generated Plan ({len(plan)} steps):")
    for step in plan:
        print(f"  Step {step['step_number']}: {step['action']} - {step['query']}")
    
    # Step 2: Execute Plan
    print(f"\n🔄 EXECUTING PLAN...")
    execution_history = []
    
    for step in plan:
        result = execute_step(step)
        execution_history.append({
            "step": step,
            "result": result
        })
        print(f"✅ Step {step['step_number']} completed")
    
    # Step 3: Synthesize Final Answer
    print(f"\n🎯 SYNTHESIZING FINAL ANSWER...")
    
    # Format execution history for the LLM
    history_text = ""
    for i, exec_item in enumerate(execution_history, 1):
        history_text += f"\nStep {i}: {exec_item['step']['action']}\n"
        history_text += f"Query: {exec_item['step']['query']}\n"
        history_text += f"Result: {exec_item['result'][:500]}...\n"  # Truncate long results
        history_text += "-" * 60 + "\n"
    
    chain = execution_prompt | llm | StrOutputParser()
    final_answer = chain.invoke({
        "execution_history": history_text,
        "original_question": query
    })
    
    return {
        "input": query,
        "plan": plan,
        "execution_history": execution_history,
        "output": final_answer
    }

# Test 1 --> Multi-step agent
query = "Who was Tamil Nadu Chief Minister during last sighting of Halley's comet"
response = run_multistep_agent(query)

print(f"\n{'#'*60}")
print(f"FINAL RESPONSE")
print(f"{'#'*60}")
print(f"\nQuery: {query}")
print(f"\nAnswer: {response['output']}")
print(f"\n{'#'*60}")