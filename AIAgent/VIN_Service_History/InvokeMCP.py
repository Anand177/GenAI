from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import asyncio, sys

async def invoke_vin_service():
    server_params= StdioServerParameters(
        command=sys.executable,
        args=["-u",
            "C:/Learning/Python/GenAI/AgenticRAG/VIN_Service_History/VinSHMCPWrapper.py"],
        env=None
    ) 

    print("Starting MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"Available Tools: {tools}")

            result= await session.call_tool(
                "vin_service_history",
                arguments={"vin" : "AAAAAAAAAAAAAAAAA"}
            )

            print(result)

            result= await session.call_tool(
                "vin_service_count",
                arguments={"vin" : "AAAAAAAAAAAAAAAAA"}
            )



asyncio.run(invoke_vin_service())
