from dotenv import load_dotenv
from langchain_core.output_parsers import NumberedListOutputParser, CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq

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
Generate 3 prime numbers between 1 and {max_number}.
Format Instructions: {format_instructions}
"""

# Define CSV Output Parser and Template
csv_op_parser = CommaSeparatedListOutputParser()
csv_fmt_instruction = csv_op_parser.get_format_instructions()
#print(format_instructions)

csv_prompt_template = PromptTemplate(
    template = template,
    input_variables = ["max_number"],
    partial_variables = {"format_instructions": csv_fmt_instruction}
)

# Create LLM Chain
#llmchain = prompt_template | groq_llm | output_parser
llmchain = RunnableSequence(
    csv_prompt_template,
    groq_llm,
    csv_op_parser
)

response = llmchain.invoke({"max_number": "30"})
print("Prime Numbers: ", response)


# Define List Output Parser and Template
lst_op_parser = NumberedListOutputParser()
lst_fmt_instruction = lst_op_parser.get_format_instructions()
#print(numbered_list_op_parser.get_format_instructions())

lst_prompt_template = PromptTemplate(
    template=template,
    input_variables = ["max_number"],
    partial_variables={"format_instructions": lst_fmt_instruction}
)

# Create LLM Chain
#llmchain = prompt_template | groq_llm | output_parser
llmchain = RunnableSequence(
    lst_prompt_template,
    groq_llm,
    lst_op_parser
)

response = llmchain.invoke({"max_number": "100"})
print("Prime Numbers: ", response)
