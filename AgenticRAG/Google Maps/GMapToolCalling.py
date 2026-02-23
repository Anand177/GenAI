from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_google_community import GooglePlacesTool
from langchain_google_genai import ChatGoogleGenerativeAI

import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
GPLACES_API_KEY=os.getenv("GCLOUD_API_KEY")
os.environ["GPLACES_API_KEY"] = os.getenv("GCLOUD_API_KEY")

llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools=[GooglePlacesTool()]       # Built-in wrapper for Google Maps

agent=initialize_agent(llm=llm,
            tools=tools,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True)

resp1=agent.run("Find good accommodation under INR 10000 nearby Sholinganallur Chennai and get distance from Ford GTBC Office")
print(resp1)

resp2=agent.run("""I'm in Tamil Nadu secretariat, need to reach Chennai Central railway station within 30 minutes. 
                Get the modes of travel available and get me the best travel mode""")
print(resp2)

resp3=agent.run("Get exact GPS Coordinates of Chennai Valluvarkottam")
print(resp3)

resp4=agent.run("Get me today's weather forecast for Vellore. Do I need to carry umbrella")
print(resp4)

resp4=agent.run("Get me today's AQI in Chennai and compare with Delhi's AQI")
print(resp4)