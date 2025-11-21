from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define Groq Model and Template
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0.7
)

template = """
Generate {number} pairs of Indian State and its area in square kilometer.
Format instructions: {format_instruction}
"""

class StateAreaCombo(BaseModel):
    state: str = Field(description="This is an Indian state")
    area: int = Field(description="Area of corresponding state in sq km")

class ListStateAreaCombo(BaseModel):
    result: list[StateAreaCombo]

pydantic_op_parser = PydanticOutputParser(pydantic_object=ListStateAreaCombo)

prompt_template = PromptTemplate(
    template=template,
    input_variables = ["number"],
    partial_variables= {"format_instruction" : pydantic_op_parser.get_format_instructions}
)
#print(prompt_template.format(number =10))

llm_chain = RunnableSequence(
    prompt_template,
    groq_llm,
    pydantic_op_parser
)

response = llm_chain.invoke(10)
print(response)