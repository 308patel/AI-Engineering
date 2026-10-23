from langchain_experimental.tools import PythonREPLTool

python_tool = PythonREPLTool()
# result = python_tool.run("print(2 + 2 * ( 4 * 3 ))")

code = """
numbers = [10, 20, 30, 40, 50]
average = sum(numbers) / len(numbers)
print("The average is:", average)
"""
result = python_tool.run(code)
print("result ::", result)