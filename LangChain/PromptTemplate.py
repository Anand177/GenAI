import random
from langchain_classic.prompts import PromptTemplate

instruction_template = " You are a {subject} teacher for {grade_level} students."

prompt_template = PromptTemplate(
    template=instruction_template,
    input_variables=["subject", "grade_level"]
)

prompt = prompt_template.format(subject="math", grade_level="5th grade")
print(prompt)

prompt = prompt_template.format(subject="science", grade_level="7th grade")
print(prompt)

# Create a  partial prompt by fixing one variable

partial_prompt = prompt_template.partial(subject="history") # Fix the subject to history
prompt = partial_prompt.format(grade_level="8th grade")
print(prompt)

# Create function partials
def generate_random_number(min_value=1, max_value=10):
    return random.randint(min_value, max_value)

partial_prompt = prompt_template.partial(grade_level=generate_random_number) 
prompt = partial_prompt.format(subject="geography")
print(prompt)

partial_prompt = prompt_template.partial(grade_level=generate_random_number) 
prompt = partial_prompt.format(subject="geography")
print(prompt)
