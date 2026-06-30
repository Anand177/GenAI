# Node has to installed

import os

from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams, StdioServerParameters

from google.genai import types

from dotenv import load_dotenv

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GCLOUD_API_KEY = os.getenv("GCLOUD_API_KEY")
os.environ["GOOGLE_MAPS_API_KEY"] = GCLOUD_API_KEY
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, max_delay=3, attempts=30)

model = Gemini(model="gemini-2.5-flash", retry_options=RETRY_OPTIONS)
session_service = InMemorySessionService()
user_id = "avasant5"

maps_toolset = McpToolset(connection_params = StdioConnectionParams(
    server_params=StdioServerParameters(command="cmd",
        # /q turns off echo/banners, /c tells it to execute the string and exit
        args=["/q", "/c", "npx", "--yes", "@modelcontextprotocol/server-google-maps"],
        env={"GOOGLE_MAPS_API_KEY": GCLOUD_API_KEY}
    )
))
maps_assistant_agent = Agent(name="maps_assistant_agent", model=model,
                             instruction="""Help the user with mapping, directions, and finding places
        using Google Maps tools.""", tools = [maps_toolset] )

app = App(name="McpAgent", root_agent= maps_assistant_agent)

