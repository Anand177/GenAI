# pip show wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool

api_wrapper = WikipediaAPIWrapper(doc_content_chars_max=50, top_k_results=3)
query="LLM"

response=api_wrapper.load(query=query)
print(response)

# Creating Tool with Wikipedia Wrapper
wiki_tool=Tool(
    name="Wikipedia",
    description="Wikipedia Search Tool",
    func=api_wrapper.load
)
print("Tool with Wikipedia Wrapper")
response=wiki_tool.invoke({"query" : query})
print(response)

# Built-in Wikipedia Tool class
#wikipedia_tool=WikipediaQueryRun(api_wrapper=api_wrapper) # With default config
wikipedia_tool=WikipediaQueryRun(api_wrapper=api_wrapper,
                    name="Anand_Wiki",
                    description="Anand's Wikipedia Search Tool")

print(f"Name -> {wikipedia_tool.name}")
print(f"Description -> {wikipedia_tool.description}")
print(f"API Wrapper -> {wikipedia_tool.api_wrapper}")
print(f"Arg Schema -> {wikipedia_tool.args_schema}")
wikipedia_tool
response=wikipedia_tool.invoke({"query" : query})
print(response)
