from langchain_core.tools import tool
from langchain_core.tools import StructuredTool

@tool
def multiply_tool(a: int, b:int) -> int:
    """
    Multiply two integers.
    Args:
        a (int): First Integer
        b (int): Second Integer
    Returns:
        int: Product of First and Second Integer

    Example:
        >>> multiply_tool(3, 2) yields 6 
    """
    return a * b

print(f"Tool Name --> {multiply_tool.name}")
print(f"Tool Description --> {multiply_tool.description}")
print(f"Args Schema --> {multiply_tool.args_schema.__dict__}")


def divide_function(a: int, b:int) -> int:
    """
    Divide two integers.
    Args:
        a (int): Dividend
        b (int): Divisor (Must not be 0)
    Returns:
        float: Quotient of dividend divided by divisor

    Example:
        >>> divide_function(6, 2) yields 3 
    """
    return a / b


division_tool = StructuredTool.from_function(divide_function)

print(f"Tool Name --> {division_tool.name}")
print(f"Tool Description --> {division_tool.description}")
print(f"Args Schema --> {division_tool.args_schema.__dict__}")

