from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableBranch

# RunnableLambda example
def add_one(x: int) -> int:
    return x + 1

add_one_runnable = RunnableLambda(add_one)
result = add_one_runnable.invoke(5)

print("Result from RunnableLambda add_one:", result)

# RunnablePassthrough example
donothing_runnable = RunnablePassthrough()
input_value = {"a": 1, "b": 2}

output_value = donothing_runnable.invoke(input_value)
print("Result from RunnablePassthrough:", output_value)

add_new_key_runnable = RunnablePassthrough.assign(c = lambda input : 3)
enhanced_output = add_new_key_runnable.invoke(input_value)

print("Result from RunnablePassthrough with new key:", enhanced_output)

# RunnableSequence
def multiply_by_two(x: int) -> int:
    return x * 2
multiply_by_two_runnable = RunnableLambda(multiply_by_two)

def subtract_three(x: int) -> int:
    return x - 3    
subtract_three_runnable = RunnableLambda(subtract_three)

runnableSeq = add_one_runnable | multiply_by_two_runnable | subtract_three_runnable
final_result = runnableSeq.invoke(4)
print("Result from RunnableSequence:", final_result)    

final_result = runnableSeq.invoke(10)
print("Result from RunnableSequence with input 10:", final_result)

runnableSeq = RunnableSequence(add_one_runnable, multiply_by_two_runnable, subtract_three_runnable)
final_result = runnableSeq.invoke(7)
print("Result from RunnableSequence (alternative) with input 7:", final_result)

runnableParrll = RunnableParallel(
    original = RunnablePassthrough(),
    added = add_one_runnable,
    multiply = multiply_by_two_runnable,
    subrtct = subtract_three_runnable
)
parrll_result = runnableParrll.invoke(5)
print("Result from RunnableParallel input 5:", parrll_result)


# Runnable branch example
# action key not found
def error_condition(x: dict) -> dict:
    if "action" not in x:
        x["error"] = "Action NOT provided"
    else:
        x["error"] = "Invalid action. Only 'add', 'multiply' & 'subtract are supported"
        
    return x

# Create a default runnable
default_runnable = RunnableLambda(error_condition)

# Takes dictionary as input and returns dict as output
def add_1_dict(x: dict) -> dict:
    x["output"] = x["number"] + 1
    return x

def multiply_2_dict(x: dict) -> dict:
    x["output"] = x["number"]*2
    return x

def subtract_3_dict(x: dict) -> dict:
    x["output"] = x["number"] - 1
    return x

# Define the branches as (condition, runnable) tuples
branch = RunnableBranch(
    (lambda input: input.get("action", "NA") == "add", RunnableLambda(add_1_dict)),
    (lambda input: input.get("action", "NA") == "multiply", RunnableLambda(multiply_2_dict)),
    (lambda input: input.get("action", "NA") == "subtract", RunnableLambda(subtract_3_dict)),
    default_runnable
)

# Change the action (add, multiply, subtract, invalid) to see the branching
branch_response = branch.invoke({ "action": "invalid", "number": 5}) 
print("Result from RunnableBranch with invalid action:", branch_response)

branch_response = branch.invoke({ "action": "add", "number": 10})
print("Result from RunnableBranch with add action:", branch_response)

branch_response = branch.invoke({ "action": "multiply", "number": 7})
print("Result from RunnableBranch with multiply action:", branch_response)