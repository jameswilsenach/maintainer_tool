"""
GitHub Issue Triage Agent
Fetches open issues via GitHub API, retrieves repo context, and uses an LLM to triage.
Follows Google Python Style Guide.
"""

import os
import sys
import re
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------

def load_environment() -> Dict[str, str]:
    """
    Loads environment variables securely. 
    """
    load_dotenv()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not found. Please check your .env file.")
        sys.exit(1)

    return {
        "GITHUB_TOKEN": token,
        "TARGET_REPO": os.getenv("TARGET_REPO"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "claude-3-haiku-20240307"),
        "GITHUB_API_URL": "https://api.github.com"
    }

# ---------------------------------------------------------
# GITHUB API INTEGRATION
# ---------------------------------------------------------

def fetch_repo_readme(repo: str, headers: Dict[str, str]) -> str:
    """
    Fetches the raw README.md file from the target GitHub repository.
    """
    url = f"https://api.github.com/repos/{repo}/readme"
    readme_headers = headers.copy()
    readme_headers["Accept"] = "application/vnd.github.v3.raw"
    
    print(f"Fetching README for {repo}...")
    try:
        response = requests.get(url, headers=readme_headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch README. Details: {e}")
        return ""

def fetch_github_issues(repo: str, headers: Dict[str, str], limit: int = 5, page: int = 1) -> List[Dict[str, Any]]:
    """
    Fetches open issues for a given repository with pagination.
    Filters out pull requests.
    """
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {
        "state": "open",
        "sort": "created",
        "direction": "desc",
        "per_page": limit,
        "page": page
    }
    
    print(f"Fetching {limit} issues from {repo} (Page {page})...")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        issues = response.json()
        
        # The GitHub API returns Pull Requests as issues, so we filter them out
        return [issue for issue in issues if "pull_request" not in issue]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}")
        sys.exit(1)

# ---------------------------------------------------------
# LIGHTWEIGHT RAG / SEARCH ENGINE
# ---------------------------------------------------------

def chunk_by_headers(markdown_text: str) -> List[Dict[str, str]]:
    """
    Splits markdown text by headers, preserving the header hierarchy.
    """
    if not markdown_text:
        return []
        
    lines = markdown_text.strip().split('\n')
    chunks = []
    current_header = "General Context"
    current_content = []
    
    for line in lines:
        if line.startswith('#'):
            if current_content:
                chunks.append({
                    "section_header": current_header, 
                    "text": " ".join(current_content).strip()
                })
                current_content = []
            current_header = re.sub(r'^#+\s*', '', line).strip()
        elif line.strip():
            current_content.append(line.strip())
            
    if current_content:
        chunks.append({
            "section_header": current_header, 
            "text": " ".join(current_content).strip()
        })
        
    return chunks

def search_repo_docs(query: str, chunks: List[Dict[str, str]], top_k: int = 2) -> List[Dict[str, str]]:
    """
    A lightweight, deterministic semantic search using term overlap.
    """
    if not chunks:
        return []
        
    query_terms = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split()) | set(chunk["section_header"].lower().split())
        overlap = len(query_terms.intersection(chunk_words))
        scored_chunks.append((overlap, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_k]]

def fetch_remote_python_chunks(repo: str, headers: Dict[str, str], max_files: int = 200) -> List[Dict[str, str]]:
    """
    Recursively indexes a REMOTE Python codebase via GitHub's API.
    Fetches raw file contents into memory and extracts functions/classes via AST.
    """
    print(f"Fetching repository tree for {repo}...")
    try:
        # 1. Get the default branch
        repo_info = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=10).json()
        branch = repo_info.get("default_branch", "main")
        
        # 2. Get the recursive tree
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        tree_data = requests.get(tree_url, headers=headers, timeout=10).json()
        
        # 3. Filter for Python files (excluding tests, docs, and hidden files)
        py_files = []
        for item in tree_data.get("tree", []):
            path = item.get("path", "")
            if path.endswith(".py") and "tests/" not in path and "docs/" not in path and not path.startswith("."):
                py_files.append(path)
                
        # Bounded behavior: Limit the number of files we download to prevent rate-limiting/hanging
        target_files = py_files[:max_files]
        if not target_files:
            return []
            
        print(f"-> Downloading and AST parsing {len(target_files)} remote Python files (Bounded by max_files={max_files})...")
        
        chunks = []
        file_headers = headers.copy()
        file_headers["Accept"] = "application/vnd.github.v3.raw" # Request raw file content
        
        # 4. Fetch and parse each file in memory with a progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Fetching & Parsing AST...", total=len(target_files))
            
            for path in target_files:
                file_url = f"https://api.github.com/repos/{repo}/contents/{path}"
                try:
                    resp = requests.get(file_url, headers=file_headers, timeout=5)
                    if resp.status_code == 200:
                        content = resp.text
                        module = ast.parse(content)
                        file_lines = content.split('\n')
                        
                        for node in module.body:
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                                start = node.lineno - 1
                                # Fallback if end_lineno isn't available in older Python versions
                                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 10
                                code_segment = "\n".join(file_lines[start:end])
                                
                                chunks.append({
                                    "section_header": f"{path} - {node.name}",
                                    "text": code_segment,
                                    "filepath": path
                                })
                except Exception:
                    # Silently skip files that fail to download or parse
                    pass
                
                # Increment the progress bar after each file
                progress.advance(task)
                
        return chunks
        
    except Exception as e:
        print(f"Warning: Remote codebase indexing failed ({e}).")
        return []

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():
    config = load_environment()
    
    parser = argparse.ArgumentParser(description="AI-powered GitHub Issue Triage Agent.")
    parser.add_argument(
        "--repo", 
        type=str, 
        default=config["TARGET_REPO"],
        help="GitHub repository (e.g., 'scverse/scanpy'). Defaults to .env TARGET_REPO."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5,
        help="Number of issues to fetch per interactive batch."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=".",
        help="Path to local repository to perform Codebase RAG. Defaults to current directory."
    )
    args = parser.parse_args()

    if not args.repo:
        print("ERROR: No repository specified. Provide --repo or set TARGET_REPO in your .env")
        sys.exit(1)

    headers = {
        "Authorization": f"token {config['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json"
    }

    print(f"\n--- Initializing Triage Pipeline for {args.repo} ---")
    
    # 1. Fetch Documentation
    readme_markdown = fetch_repo_readme(args.repo, headers)
    doc_chunks = chunk_by_headers(readme_markdown)
    print(f"-> Parsed README into {len(doc_chunks)} searchable architectural chunks.")
    
    # 1.5 Fetch Remote Codebase
    code_chunks = fetch_remote_python_chunks(args.repo, headers, max_files=200)
    print(f"-> Indexed {len(code_chunks)} remote Python functions/classes for Codebase RAG.")
    
    # 2. Interactive Issue Processing Loop
    page = 1
    while True:
        print(f"\n=== Fetching Batch {page} ===")
        issues = fetch_github_issues(args.repo, headers, limit=args.chunk_size, page=page)
        
        if not issues:
            print("\n[success] No more active issues found. Triage complete![/success]")
            break
            
        print(f"-> Retrieved {len(issues)} active issues for triage.")
        
        # 3. Quick test of our RAG mapping logic on this batch
        for i, sample_issue in enumerate(issues, 1):
            issue_text = f"{sample_issue['title']} {sample_issue.get('body', '')}"
            print(f"\n[Test] Issue {i}/{len(issues)}: #{sample_issue['number']} - {sample_issue['title']}")
            
            # Doc RAG
            relevant_docs = search_repo_docs(issue_text, doc_chunks, top_k=1)
            if relevant_docs:
                print(f"  -> Docs Match: '{relevant_docs[0]['section_header']}'")
            else:
                print("  -> Docs Match: None")
                
            # Code RAG
            relevant_code = search_repo_docs(issue_text, code_chunks, top_k=1)
            if relevant_code:
                print(f"  -> Code Match: '{relevant_code[0]['section_header']}' (in {Path(relevant_code[0]['filepath']).name})")
            else:
                print("  -> Code Match: None")
        
        # 4. Interactive Prompt
        print("\n" + "-"*50)
        user_input = input(f"Press [Enter] to fetch the next {args.chunk_size} issues, or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            print("Exiting triage process. Have a great Monday!")
            break
            
        page += 1

if __name__ == "__main__":
    main()