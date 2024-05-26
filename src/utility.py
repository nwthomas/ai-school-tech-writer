from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from github import Github
from .constants import *
from typing import Any, List
import base64
import json
import os

# Initialize PyGithub, fetch repo, and fetch pull request
g = Github(GITHUB_API_KEY)
repo = g.get_repo(REPO_PATH)
pull_request = repo.get_pull(PULL_REQUEST_NUMBER)

@tool
def leave_code_review_comment(comment: str, filename: str, line: int) -> None:
    """
    Leave a code review comment on a pull request at a specified file and line.

    Parameters:
    comment (str): The comment to leave as a code review comment
    filename (str): The filename to leave a code review comment on
    line (int): The line in the file to leave a code review comment on
    """
    print("TRYING TO LEAVE CODE COMMENT", comment, filename, line)
    pull_request.create_review_comment(response.comment, diff.filename, response.line)

def load_documents(repo: Any) -> List[str]:
    """Loads PDF documents to be used in embeddings in a vector store"""
    repo.get_branch(BASE_BRANCH)
    loader = DirectoryLoader("./", glob="*.*")
    raw_documents = loader.load()

    return raw_documents

def get_split_documents(raw_documents: List[str]) -> List[str]:
    """Chunks codebase documents to be used in embeddings in a vector store"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_documents = text_splitter.split_documents(raw_documents)

    return split_documents

def embed_documents(repo: Any, index_name: str) -> None:
    """Embeds chunked documents in Pinecone's vector store after creating a new index"""
    raw_documents = load_documents(repo)
    split_documents = get_split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

    PineconeVectorStore.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name=index_name
    )

def get_embeddings_for_diffs(index_name: str, diffs: List[str]) -> List[str]:
    """Takes a list of diffs from a PR, stringifies them, and does embedded search"""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    document_vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = document_vectorstore.as_retriever()

    json_pull_request_diffs = json.dumps(diffs)

    return retriever.invoke(json_pull_request_diffs)

def delete_embeddings_for_codebase(index_name: List[str]) -> str:
    """Deletes an index from the Pinecone account"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pc.delete_index(index_name)

def format_data_for_description_prompt(diffs, commit_messages, codebase_context, pull_request_description_template):
    """Formats contextual data and generates the prompt with it, including for system data"""

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

def generate_pr_description(prompt: str) -> str:
    """Uses a model (currently OpenAI) to generate a new pull request description"""
    client = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

    messages = [
        {"role": "system", "content": "You are an expert code reviewer tasked with summarizing all changes in a pull request and writing an incredible description for it."},
        {"role": "user", "content": prompt}
    ]

    response = client.invoke(input=messages)
    parser = StrOutputParser()
    content = parser.invoke(input=response)

    return content

def update_pr_description(repo: Any, pull_request_number: int, pull_request_description: str) -> None:
    """Updates a given pull request at a given repo with a new description"""
    pr = repo.get_pull(pull_request_number)
    pr.edit(body=pull_request_description)

def format_data_for_code_review_prompt(diff, codebase_context) -> List[str]:
    # Construct the prompt with clear instructions for the LLM
    prompt = (
        "Here's some code I wrote:\n"
        f"{diff}\n"
        "Here's context on it from existing files in the codebase:"
        f"{codebase_context}"
        """
        You are an expert software engineer and are known on your team for being a thorough and precise code reviewer. Review the code diff I just gave you along with the codebase context. If you spot code that needs to change or improve in some way, return comments on it in this format:

        [
            {'content': 'Consider using a more descriptive name for the job and steps to make the workflow easier to understand and maintain.', 'line': 42},
            {'content': 'Consider adding error handling for the GitHub API calls to handle potential failures gracefully.', 'line': 3},
        ]

        Always return a list at root. Nest dictionaries inside it with keys of content (for your comment) and line (for the line to leave the comment on - you can get this line from the code review diffs). Use double quotes and never use single quotes for enclosing keys or values in the dictionaries.

        Only return JSON like I've described. 
        
        Do not EVER return standalone markdown text or any text content. For example, never return a bulleted list or markdown headers in a long description file
        
        Return comments (one or multiple) in a list like the example. Keep the comments short with no yapping. Try to eliminate whole classes of errors, suggest cleaner code patterns, and predict issues that may happen if your teammate's code lands in production.\n
        """
    )

    return prompt

def run_code_review_for_diff(prompt):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")
    return llm.invoke(prompt)
