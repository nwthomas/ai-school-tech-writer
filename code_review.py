from github import Github
from src.constants import *
from src.utility import *
from typing import List
import jsonschema
import random
import json
import sys

def generate_code_review():
    # Initialize PyGithub, fetch repo, and fetch pull request
    g = Github(GITHUB_API_KEY)
    repo = g.get_repo(REPO_PATH)
    pull_request = repo.get_pull(PULL_REQUEST_NUMBER)

    # Get all current pull request diffs
    pull_request_diffs = [
        {
            "filename": file.filename,
            "patch": file.patch,
        }
        for file in pull_request.get_files()
    ]

    print(pull_request_diffs)

    # # Dynamic vector database index name
    # random_number = random.randint(1, 100000000)
    # current_index_name = f"auto-code-review-{random_number}"
    # embed_documents(repo, current_index_name)

    # # Search embeddings
    # for diff in pull_request_diffs:
    #     codebase_context = get_embeddings_for_diffs(current_index_name, [diff])

    #     # Prompt LLM with context + diff to decide if a comment is necessary
    #     prompt = format_data_for_code_review_prompt(
    #         diff,
    #         codebase_context,
    #     )

    #     result = run_code_review_for_diff(prompt)

    #     schema = {
    #         "type": "array",
    #         "items": {
    #             "type": "object",
    #             "properties": {
    #                 "content": { "type": "string" },
    #                 "line": { "type": "integer" },
    #             },
    #             "required": ["content", "line"],
    #         },
    #     }

    #     print(diff["filename"], len(codebase_context))

    #     try:
    #         json_data = json.loads(result.content)
    #         jsonschema.validate(instance=json_data, schema=schema)
    #         print("NEW COMMENT SECTION\n\n\n" + result.content + "\n\n\n")
    #         # pull_request.create_review_comment(json_data.content, diff.filename, json_data.line)
        
    #     except jsonschema.ValidationError as e:
    #         print("JSON validation error:", e.message, result.content)
    #     except json.decoder.JSONDecodeError as e:
    #         print("Invalid JSON format:", e, result.content)
    #     except Exception as e:
    #         print("An error occurred:", e, result.content)

    # # Delete index and codebase embeddings in vector database
    # delete_embeddings_for_codebase(current_index_name)

if __name__ == '__main__':
    generate_code_review()
