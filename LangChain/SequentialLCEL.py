import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
HF_API_KEY=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1.Define a LLM for term definition
term_llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.5,
    top_p=0.9,
    top_k=40,
)

# 2. Define template and prompt for term definition
term_template = """
Define the given term in very creative manner under 80 words.

Term: {term}
"""
term_prompt = PromptTemplate(
    template=term_template,
    input_variables=["term"]
)

# 3. Create chain for term definition
term_chain = term_prompt | term_llm

# 4. Define a LLM for scoring the definition
score_llm = GoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.5
)

# 5. Define template and prompt for scoring the definition
score_template = """
Given definition of a term, rate definition on scale of 0 to 5 based on accuracy and completeness.
With 0 being lowest and 5 being highest. Explanation for each parameter should not exceed 20 words.
Definition: {definition}
"""

score_prompt = PromptTemplate(
    template=score_template,
    input_variables=["definition"]
)

# 6. Create chain for scoring the definition
score_chain = score_prompt | score_llm

# 7. Combine both chains in sequence
seq_chain = {"definition" : term_chain} | score_chain

response = seq_chain.invoke({"term": "LangChain"})
print("Final Score Response:\n", response)

response = seq_chain.invoke({"term": "Gen AI"})
print("Final Score Response:\n", response)