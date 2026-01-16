from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

template = ChatPromptTemplate.from_messages(
    [SystemMessage(content="You are a sarcastic comedian"),
     ("human", "Tell me a joke about {topic}")]
)

chat_llm=init_chat_model(model="gpt-4o-mini", model_provider="openai")

messages=template.format_messages(topic="car")
response=chat_llm.invoke(messages)

print(response.content)
