import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_classic.chains.prompt_selector import ConditionalPromptSelector

from langchain_community.llms.ai21 import AI21
from langchain_community.llms.cohere import Cohere

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")


default_prompt = """Summazire the following text in a concise manner.

Text: {text}
"""

default_prompt_template = PromptTemplate(template=default_prompt, input_variables=["text"] )

ai21_prompt = """
<<SYS>>
You are an assistant tasked with summarizing technical research papers.
<</SYS>>

[INST]
Generate Summary:

{text}
[/INST]"""

ai21_prompt_template = PromptTemplate(template=ai21_prompt, input_variables=["text"])
ai21_llm = AI21(ai21_api_key=os.getenv("AI21_API_KEY"))

cohere_prompt = """
You are a helpful assistant. Your task is to summarize the given content:

Content: 
{text}
"""

cohere_prompt_template = PromptTemplate(template=cohere_prompt, input_variables=["text"])
cohere_llm = Cohere(cohere_api_key=os.getenv("COHERE_API_KEY"))


conditional_prompt_selector = ConditionalPromptSelector(
    default_prompt= default_prompt_template,
    conditionals=[
        (lambda llm: isinstance(llm, type(ai21_llm)), ai21_prompt_template),
        (lambda llm: isinstance(llm, type(cohere_llm)), cohere_prompt_template),
    ]
)

llm=ai21_llm  # or cohere_llm   
prompt= conditional_prompt_selector.get_prompt(llm)
print(prompt.format(text="Text to summarize"))


llm=cohere_llm
prompt = conditional_prompt_selector.get_prompt(llm)
print(prompt.format(text="Text to summarize"))
llm.invoke( prompt.format(text="Text to summarize") )
