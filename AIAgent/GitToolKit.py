# pip install pygithub
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_core.tools import BaseTool
from typing import List


# Print and Retrieve List of Tools in GitHub Toolkit
def list_github_tools(toolkit : GitHubToolkit):
    
    tools : List[BaseTool] = toolkit.get_tools()
    tool_names = []
    print(f"Available Tool # -> {len(tools)}")

    for tool in tools:
        tool_names.append(tool.name)
        print("-"*80)
        print(f"Name -> {tool.name}")
        print(f"Description -> {tool.description}")
    return tool_names


# Generic Function to find and execute tool
def run_tool(toolkit : GitHubToolkit, tool_name: str, tool_input):

    tools : List[BaseTool] = toolkit.get_tools()
    target_tool = next(tool for tool in tools if tool.name == tool_name)
    result = target_tool.run(tool_input)

    print("-"*80)
    print(f"Tool Name -> {tool_name}")
    print(f"Input -> {tool_input}")
    print(f"Output -> {result}")
    print("-"*80)


# Get files from a directory in Git repo
def git_get_repos(toolkit: GitHubToolkit):
    tool_name="Get files from a directory"
    directory="EmbeddingAndVectorDB"        # Existing directory in given Repo
    run_tool(toolkit, tool_name, directory)


# Search issues and PR based on PR query
def git_search_issues_pr(toolkit: GitHubToolkit):
    tool_name="Search issues and pull requests"
    query="is:pr author:@me"
    run_tool(toolkit, tool_name, query)


# Get Pull Request baased on PR #
def git_get_pr(toolkit: GitHubToolkit):
    tool_name="Get Pull Request"
    pr_num=1
    run_tool(toolkit, tool_name, {"pr_number": pr_num})


# Overview of files included in PR #
def git_pr_overview(toolkit: GitHubToolkit):
    tool_name="Overview of files included in PR"
    pr_num=1
    run_tool(toolkit, tool_name, {"pr_number": pr_num})


# List files in PR
def git_pr_file(toolkit: GitHubToolkit):
    tool_name="List Pull Requests' Files"
    pr_num=1
    run_tool(toolkit, tool_name, {"pr_number": pr_num})


# Read a file from Git repo
def git_read_file(toolkit: GitHubToolkit):
    tool_name="Read File"
    file="EmbeddingAndVectorDB/ChromaDB.py"
    run_tool(toolkit, tool_name, file)


# Overview of files in current working branch. No input query needed
def git_get_file_cwb(toolkit: GitHubToolkit):
    tool_name="Overview of files in current working branch"
    run_tool(toolkit, tool_name, "")


# List branches in Git repository. No input query Needed
def git_list_branch(toolkit: GitHubToolkit):
    tool_name="List branches in this repository"
    run_tool(toolkit, tool_name, "")


def main():

    github_app_id="2417019"
    github_repository="Anand177/GenAI"
    github_private_key=""

    git_private_Key_file="C:/Learning/AI/Key/anandgitlangchainagent.2025-12-05.private-key.pem"
    try:
        with open(git_private_Key_file, 'r') as file_obj:
            github_private_key=file_obj.read()
#            print(file_content)
    except Exception as e:
        print(f"Exception -> {e}")
        exit

    print(f"GitHub App Id : {github_app_id}")
    print(f"GitHub Repository : {github_repository}")
    print(f"Git Private Key Len : {len(github_private_key)}")

    github_wrapper = GitHubAPIWrapper(
        github_app_id=github_app_id,
        github_repository=github_repository,
        github_app_private_key=github_private_key
    )

    github_toolkit = GitHubToolkit.from_github_api_wrapper(github_wrapper)
    list_github_tools(github_toolkit)


    # Execute Tools
    git_get_repos(github_toolkit)
    git_search_issues_pr(github_toolkit)
    git_read_file(github_toolkit)
    git_get_file_cwb(github_toolkit)
    git_list_branch(github_toolkit)
    git_get_pr(github_toolkit)
    git_pr_overview(github_toolkit)
    git_pr_file(github_toolkit)





if __name__ == "__main__":
    main()