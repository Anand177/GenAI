from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import PromptTemplate

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini",
            temperature=0.5,
            top_p=0.7,
            api_key=OPENAI_API_KEY,
            frequency_penalty=1
)

#print(llm)

query = "Who is the PM of United Kingdom."

response =llm.invoke(query)
print(response.content)

## Retrieve data from Wiki
def get_context(topic):
    wiki_retriever=WikipediaRetriever()
    docs=wiki_retriever.invoke(topic)
    context=''
    for doc in docs:
        context = context + doc.page_content + "\n"

    return context

## Setup prompt
prompt = PromptTemplate(
    template="""You are a smart agent who only use provided context to carry out given task
    Task: {task}
    Context: {context}
    """,
    input_variables=["task", "context"]
)

topic="Prime Minister of the United Kingdom"
context=get_context(topic)
#print(context)

result=llm.invoke(prompt.format(task=query, context=context))

print(result.content)