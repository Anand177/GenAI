from dotenv import load_dotenv
from langchain_tavily.tavily_search import TavilySearch

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

tavily_tool = TavilySearch(include_raw_content=True, 
                                  include_answer=False, # Not including AI generated response
                                  include_images=False,
                                  search_depth="basic", 
                                  max_results=3)

query="Who is Anand vasantharajan?"
results=tavily_tool.run(query)

print(results)