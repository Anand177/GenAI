from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.tools import BaseTool
from typing import List

import os
import time

local_dir="C:/Learning/LocalFileMgmtToolDir"
base_file="BaseFile.txt"
base_file_content=f"This file is created by User -> {os.getlogin()} at {int(time.time())}"

copied_file=f"CopiedFile_{int(time.time())}.txt"
moved_file=f"MovedFile_{int(time.time())}.txt"

# Print and Retrieve List of File management Tools
def list_file_mgmt_tools(toolkit : FileManagementToolkit):
    
    tools : List[BaseTool] = toolkit.get_tools()
    tool_names = []
    print(f"Available Tool # -> {len(tools)}")

    for tool in tools:
        tool_names.append(tool.name)
        print("-"*80)
        print(f"Name -> {tool.name}")
        print(f"Description -> {tool.description}")
    return tool_names


# Generic Method to run File Management Tools
def run_tool(toolkit : FileManagementToolkit, tool_name: str, tool_input):

    tools : List[BaseTool] = toolkit.get_tools()
    target_tool = next(tool for tool in tools if tool.name == tool_name)
    result = target_tool.run(tool_input)

    print("-"*80)
    print(f"Tool Name -> {tool_name}")
    print(f"Input -> {tool_input}")
    print(f"Output -> {result}")
    print("-"*80)

#write_file
# Create/Write File
def write_file_tool(toolkit : FileManagementToolkit):
    tool_name="write_file"
    write_input = {
        "file_path" : base_file,
        "text" : base_file_content
    }
    run_tool(toolkit, tool_name, write_input)
    

# Read File
def read_file_tool(toolkit : FileManagementToolkit):
    tool_name="read_file"
    read_input = {"file_path" : base_file}
    run_tool(toolkit, tool_name, read_input)


# Copy File
def copy_file_tool(toolkit : FileManagementToolkit):
    tool_name="copy_file"
    copy_input = {
        "source_path" : base_file,
        "destination_path" : copied_file
    }
    run_tool(toolkit, tool_name, copy_input)


# Move File
def move_file_tool(toolkit : FileManagementToolkit):
    tool_name="move_file"
    move_input = {
        "source_path" : base_file,
        "destination_path" : moved_file
    }
    run_tool(toolkit, tool_name, move_input)


#Search for files with given pattern
def file_search_tool(toolkit : FileManagementToolkit):
    tool_name="file_search"
    search_input = {
        "dir_path" : ".",
        "pattern" : "*Copied*"
    }
    run_tool(toolkit, tool_name, search_input)


# List Directory
def list_dir_tool(toolkit : FileManagementToolkit):
    tool_name="list_directory"
    list_input = {"dir_path" : "."} #Current Dir
    run_tool(toolkit, tool_name, list_input)


# Delete File
def del_file_tool(toolkit : FileManagementToolkit):
    tool_name="file_delete"
    delete_input = {"file_path": copied_file}
    run_tool(toolkit, tool_name, delete_input)


def main():

    if not os.path.exists(local_dir):
        print(f"Directoy doesn't exist -> {local_dir}")
        os.makedir(local_dir)
        print(f"Directory created -> {local_dir}")
    else:
        print(f"Directory exists -> {local_dir}")

    file_mgmt_tk=FileManagementToolkit(root_dir=local_dir)
    list_file_mgmt_tools(file_mgmt_tk)


    write_file_tool(file_mgmt_tk)
    read_file_tool(file_mgmt_tk)
    copy_file_tool(file_mgmt_tk)
    move_file_tool(file_mgmt_tk)
    file_search_tool(file_mgmt_tk)
    list_dir_tool(file_mgmt_tk)
    del_file_tool(file_mgmt_tk)
    list_dir_tool(file_mgmt_tk)
    


if __name__ == "__main__":
    main()