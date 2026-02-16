from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_google_genai import GoogleGenerativeAI
from langchain_tavily.tavily_search import TavilySearch


import json
import os
import time

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

# Create Gemini LLM
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=GoogleGenerativeAI(model="gemini-flash-latest",
                    google_api_key=GOOGLE_API_KEY,
                    temperature = 0.3,
                    top_p = 0.85
)

# Create Tavily Web Search Tool
tavily_tool = TavilySearch(include_raw_content=True, 
                                  include_answer=False, # Not including AI generated response
                                  include_images=False,
                                  search_depth="basic", 
                                  max_results=3)

# Create Wikipedia Tool
wikipedia_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Create Tools
tools_json = [
    {
        "name" : "wikipedia",
        "description" : wikipedia_tool.description,
        "arguments" : [
            {"input" : "input"}
        ],
        "response" : "search results"
    },
    {
        "name" : "tavily",
        "description" : tavily_tool.description,
        "arguments" : [
            {"input" : "input"}
        ],
        "response" : "search results"
    }
]
tools = json.dumps(tools_json, indent=4)

tools_map = {   #Tools Map for Agent to map name to function
    "wikipedia" : wikipedia_tool.invoke,
    "tavily" : tavily_tool.invoke
}

print(tools)

prompt_template = """
You are a helpful assistant capable of answering questions on various topics. 
Do not use your internal knowledge. Must validate answer using tools to access external tools.

Instructions:
Take an iterative approach. In each iteration:
1. Thought : Think step by step. Use alternatives as needed.
2. Action : Execute appropriate tools. Try alternative tool if an appropriate response is not received.
3. Observations : Make observations based on the responses from the tools and decide the next step

Repeat this Thought/Action/Observation process N times till either answer is derived or unable to get answer with available tools.

Use only the following available tools to find information.

Tools Available:
{tools}

Guidelines for Responses:
Format 1: If the question cannot be answered with ANY of the available tools, use this format. 
'actions' contain alternate tools to be run:
{{
    "answer": "No appropriate tool available",
    "actions": [
        {{
            "action": tool name,
            "arguments": dictionary of argument values
        }}
    ],
    "scratchpad": {{
        "thought": "your inner thoughts",
        "action": "tool name",
        "observations": "your observations from the tool responses"
    }}
}}

Format 2: If you need to run tools to obtain the information, use this format:
{{
    "actions": [
        {{
            "action": tool name,
            "arguments": dictionary of argument values
        }}
    ],
    "scratchpad": {{
        "thought": "your inner thoughts",
        "action": "tool name",
        "observations": "your observations from the tool responses"
    }}
}}

Format 3: If you can answer the question using the responses from the tools, use this format:
{{
    "answer": "your response to the question",
    "scratchpad": {{
        "thought": "your inner thoughts",
        "action": "tool name",
        "observations": "your observations from the tool responses"
    }}
}}

Do not run a tool again with same arguments, use tool responses from previous runs of the tool.

Avoid any preamble; respond directly using one of the specified JSON formats.
Question:

{question}
Tool Responses:

Your response:
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=["tools", "question"])


def invoke_llm_with_tools(input):
    
    try:
        response = llm.invoke(input)
        print(f"LLM Response --> {response}")

        response_cleaned=response.strip()
        if response_cleaned.startswith("```json"):
            response_cleaned = response_cleaned.replace("```json", "").replace("```", "").strip()
        elif response_cleaned.startswith("```"):
            response_cleaned = response_cleaned.replace("```", "").strip()

        # Parsing JSON
        try:
            response_json=json.loads(response_cleaned)
    #response_json = response
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Response was: {response_cleaned[:200]}")
            # Return a default structure to continue
            return {
                "answer": "Error: LLM returned invalid JSON format",
                "scratchpad": {"thought": "JSON parsing failed"}
            }, {"answer": "Error: LLM returned invalid JSON format"}

        tool_responses=[]

        if "actions" in response_json and len(response_json["actions"]) > 0:
            # LLM suggested tools to be executed
            tool_responses = {"action_responses" : invoke_tools(response_json)}
        elif "answer" in response_json:     # Answer already received
            return response_json, {"answer" : response_json["answer"]}
        
        return response_json, tool_responses
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "scratchpad": {"thought": "Exception occurred"}
        }, {"answer": f"Error: {str(e)}"}

# utility function to invoke tools
def invoke_tools(llm_response):
    action_responses = []
    if len(llm_response["actions"]) == 0:
        print("No actions need to be run")
    else:
        # Run all tools suggested by LLM
        for action in llm_response["actions"]:
            action_function = tools_map[action["action"]]   # Get function pointer
            action_invoke_result = action_function(**action["arguments"])
            action["action_response"] = str(action_invoke_result)[:1000]

            action_responses.append(action)
    return action_responses


max_itr=5           # Max 4 iterations permitted
max_exec_time=60000 # Max exec time set to 1 min to prevent runaways

def multi_step_agent_loop(question: str):

    itr_num=1

    #Input from prompt template
    input =prompt.format(tools=tools, question=question)

    start_time_millis=time.time()*1000

    while itr_num <= max_itr:
        print(f"Iteration --> {itr_num}")
        print("-"*50)

        llm_response, action_response = invoke_llm_with_tools(input)

        # Check for answer attribute in response
        if "answer" in llm_response and "actions" not in llm_response:
            # Answer found, no further action needed
            answer = action_response["answer"]
            return "ANSWER", input, answer
        elif len(action_response)==0:
            # No appropriate tool found
            return "NO APPROPRIATE TOOL FOUND", input, {}
        else:
            # Argument input with 
            # 1. Response from LLM
            # 2. Response from tools
            # 3. Output Indicator {Your response: }
            input = input + "\n" + json.dumps(llm_response) + "\nTool responses:\n" + json.dumps(action_response) + "\nYour Response: "
            
        itr_num = itr_num +1
        current_time = round(time.time()*1000)

        if (current_time-start_time_millis) > max_exec_time:
            return "MAX_EXECUTION_TIME_EXCEEDED", input, {}
            
    return "MAX_EXECUTION_TIME_EXCEEDED", input, {}


question="Which team was the cricket 50 over World Cup champion when Virat Kohli was born"

stop_reason, input, answer = multi_step_agent_loop(question)

print("Answer : ", answer)
print("Stop reason : ", stop_reason)