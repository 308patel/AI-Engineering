from langchain_community.tools import ReadFileTool, WriteFileTool
from langchain_community.agent_toolkits import FileManagementToolkit

toolkit = FileManagementToolkit(root_dir=".")
tools = toolkit.get_tools()

write_tool = WriteFileTool(root_dir='.')
write_tool.run({
    "file_path": "test.txt",
    "text":"This is a test file created by the WriteFileTool."
})

read_tool = ReadFileTool(root_dir=".")
file_content = read_tool.run("test.txt")

print("File content:", file_content)
