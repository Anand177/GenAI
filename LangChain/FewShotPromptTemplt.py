from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector


examples = [
    {   "question" : "A baker made 8 chocolate chip cookies and 6 oatmeal cookies. How many cookies did he make in total",
        "answer" : "The baker made a total of fourteen cookies. ($8 + 6 = 14$)"
    },
    {   "question" : "If you have 12 apples and you eat 4 of them, how many apples do you have left?",
        "answer" : "You have eight apples left. ($12 - 4 = 8$)"
    },
    {
        "question" : "What is the result when you multiply the number four by the number three?",
        "answer" : "The result is twelve. ($4 \times 3 = 12$)"
    },
    {
        "question" : "If you have 20 candies and you share them equally among 4 friends, how many candies does each friend get?",
        "answer" : "Each friend gets five candies. ($20 \div 4 = 5$)"
    }
]


# Create template for examples
examples_template = """
Example Question: {question}
Example Answer: {answer}
"""

prompt_template_example = PromptTemplate(
    input_variables=["question", "answer"],
    template=examples_template
)

# Create few shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=prompt_template_example,
    examples=examples,
    input_variables=["input"],
    suffix="Question: {input}",
    prefix="Answer the question based on the examples provided:\n\n"
)

ques = "If you have 15 marbles and you give 5 to your friend, how many marbles do you have left?"

final_prompt = few_shot_prompt.format(input=ques)
print(final_prompt)

#Length based Example Selector

max_length = 80  # Define maximum length for selected examples
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=prompt_template_example,
    max_length=max_length
)

length_based_example_selector = FewShotPromptTemplate(
    example_prompt=prompt_template_example,
    input_variables=["input"],
    example_selector=example_selector,
    suffix="Question: {input}",
    prefix="Answer the following questions based on the examples provided:\n"
)

final_prompt = length_based_example_selector.format(input=ques)
print(final_prompt)



from huggingface_hub import InferenceClient
import os

HF_API_KEY=os.getenv("HUGGINGFACEHUB_API_TOKEN")
client = InferenceClient(api_key=HF_API_KEY)

msg = [{"role": "user", "content": final_prompt}]

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=msg,
    temperature=0.7
)
for choice in completion.choices:
    print("Message Content: ", choice.message.content)
    print("Message Reasoning: ", choice.message.reasoning)
