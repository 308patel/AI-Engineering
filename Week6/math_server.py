from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """add two numbers"""
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """subtract value of b from a numbers"""
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """multiply two numbers""" 
    return a * b
mcp.run(transport="stdio")