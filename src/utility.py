from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai import ChatOpenAI
from .constants import *
import base64
import os

def store_codebase_embeddings():
    pass

def format_data_for_prompt(diffs, commit_messages, codebase_context, pull_request_description_template):
    # Combine the changes into a string with clear delineation.
    changes = '\n'.join([
        f'File: {file["filename"]}\nDiff: \n{file["patch"]}\n'
        for file in diffs
    ])

    # Combine all commit messages
    commit_messages = '\n'.join(commit_messages) + '\n\n'

    # Construct the prompt with clear instructions for the LLM.
    prompt = (
        "Please review the following code changes and commit messages from a GitHub pull request:\n"
        "Code changes from Pull Request:\n"
        f"{changes}\n"
        "Commit messages:\n"
        f"{commit_messages}"
        "Here's context from existing files in the codebase:"
        f"{codebase_context}"
        "Consider the code changes, commit messages, and codebase context given above, write a pull request description. Be concise with no yapping, and just summarize bullet points of what's changed instead of listing all file paths and what changed inside each of them. Keep all of this relatively high level.\n"
        "Here is the markdown pull request template. Only use the sections in this content and do not add any others. Do not add a new Pull Request Description header. Keep the markdown heirarchy as it is in the content you've been given, and don't add any other headers. Do not add a new title for the template.\n"
        f"{pull_request_description_template}\n"
        "Pull request description:\n"
    )

    return prompt

def generate_pr_description(prompt):
    # Initialize model
    client = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

    try:
        messages = [
            {"role": "system", "content": "You are an expert code reviewer tasked with summarizing all changes in a pull request and writing an incredible description for it."},
            {"role": "user", "content": prompt}
        ]

        response = client.invoke(input=messages)
        parser = StrOutputParser()
        content = parser.invoke(input=response)

        return content
    except Exception as e:
        print(f'Error making LLM call: {e}')

def update_pr_description(repo, pull_request_number, pull_request_description):
    # Get the repo pull request
    pr = repo.get_pull(pull_request_number)

    # Edit the PR description
    pr.edit(body=pull_request_description)