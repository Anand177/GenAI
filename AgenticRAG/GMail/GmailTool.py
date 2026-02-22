from dotenv import load_dotenv

from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import build_resource_service, get_gmail_credentials
from langchain_google_genai import ChatGoogleGenerativeAI

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

gmail_credentials =get_gmail_credentials(token_file="C:\\Learning\\AI\\Key\\token.json",
            scopes=["https://www.googleapis.com/auth/gmail.modify"],
            client_secrets_file="C:\\Learning\\AI\\Key\\GAPI_OAuth.json")
gmail_toolkit=GmailToolkit(api_resource=build_resource_service(credentials=gmail_credentials))

for tool in gmail_toolkit.get_tools():
    print(f"Name ->{tool.name}")
    print(f"Desc -> {tool.description}")
    print(f"Args -> {tool.args_schema}")
    print(f"Meta -> {tool.metadata}")


llm=ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

agent=initialize_agent(tools=gmail_toolkit.get_tools(),
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True)

resp1 = agent.run("List Sender and Subject of all Unread mails in Inbox")
print(resp1)

resp2 = agent.run("""Summarize last 5 mails received from anand.vasantharajan@gmail.com.
Summarization for each mail shouldn't be more than 3 sentences""")
print(resp2)

resp3 = agent.run("""Check for unread email from anand.vasantharajan@gmail.com.
If present, compose response from Automobile expert perspective. 
Mention you don't know for non automobile questions.
Just compose your response and don't send mail""")
print(resp3)