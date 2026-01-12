# Refer SubGraph.ipynb
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

import random

class ParentGraph(TypedDict):       # Parent Graph
    topic: str          # Dummy Report Topic
    report: str         # Generated Content of Topic
    quality: float      # Score between 1 -> 5

class ResearchGraph(TypedDict):     # Research Subgraph sharing keys with parent
    topic: str
    report: str         # Shared variables

def research_node(state: ResearchGraph):    # Dummy Processing
    return {"report" : f"This is a Dummy report on {state['topic']}"}

# Create Research Sub graph
research_subgraph=StateGraph(ResearchGraph)

research_subgraph.add_node("research_node", research_node)

research_subgraph.add_edge(START, "research_node")
research_subgraph.add_edge("research_node", END)

research_subgraph_compiled=research_subgraph.compile()
#response=research_subgraph_compiled.invoke({"topic" : "Cricket"})
#print(response)

class ReviewerGraph(TypedDict):     # Reviewer Sub graph not sharing keys with parent
    content: str
    score: int

def review_node(state: ReviewerGraph):      # Return dummy score
    return {"score" : random.randint(1,5)}

review_subgraph=StateGraph(ReviewerGraph)

review_subgraph.add_node("review_node", review_node)

review_subgraph.add_edge(START, "review_node")
review_subgraph.add_edge("review_node", END)

review_subgraph_compiled=review_subgraph.compile()
#response=review_subgraph_compiled.invoke({"article" : "dummy"})
#print(response)

#Parent Node
def topic_node(state: ParentGraph):
    return ({"topic" : "movie"})

# Method to invoke Review Subgraph 
# Convert state schema to match with sub graph's schema. Invoked through Node
def call_review_node(state: ParentGraph):
    response=review_subgraph_compiled.invoke({"content": state["report"]})
    review_score=response['score']
    return {"quality" : review_score}

parent_graph= StateGraph(ParentGraph)

parent_graph.add_node("topic_node", topic_node)
parent_graph.add_node("research_node", research_subgraph_compiled)
parent_graph.add_node("call_review_node", call_review_node)

parent_graph.add_edge(START, "topic_node")
parent_graph.add_edge("topic_node", "research_node")
parent_graph.add_edge("research_node", "call_review_node")
parent_graph.add_edge("call_review_node", END)

parent_graph_compiled = parent_graph.compile()
response=parent_graph_compiled.invoke({})

print(response)