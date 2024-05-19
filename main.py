import os
from github import Github
from utility import *

github_api_key = os.getenv("GITHUB_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
repo_path = os.getenv('REPO_PATH')
pull_request_number = int(os.getenv('PR_NUMBER'))

def main():
    repo = g.get_repo(repo_path)
    pr_description_template = repo.get_contents("./.github/PULL_REQUEST_TEMPLATE.md")
    pull_request = repo.get_pull(pull_request_number)

    pull_request_diffs = [
        {
            "filename": file.filename,
            "patch": file.patch 
        } 
        for file in pull_request.get_files()
    ]

    commit_messages = [commit.commit.message for commit in pull_request.get_commits()]
    prompt = format_data_for_openai(pull_request_diffs, readme_content, commit_messages)
    updated_readme = call_openai(prompt)
    update_readme_and_create_pr(repo, updated_readme, readme_content.sha)

if __name__ == '__main__':
    main()
