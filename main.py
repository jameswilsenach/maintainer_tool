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

# --- NEW IMPORTS FOR LLM & SCHEMAS ---
from pydantic import BaseModel, Field
import litellm
import instructor
import time
import logfire
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

# ---------------------------------------------------------
# DATA SCHEMAS
# ---------------------------------------------------------

class IssueDigest(BaseModel):
    """
    Pydantic schema for digesting and extracting key information from a raw GitHub issue.
    This structured data drives the downstream RAG retrieval.
    """
    key_phrases: List[str] = Field(
        description="A list of 3-5 specific keywords or phrases relevant to the issue (e.g., 'Leiden', 'clustering', 'modularity'). These will be used as semantic search queries."
    )
    code_reasoning: str = Field(
        description="Reasoning that primes the agent to extract relevant code chunks or single functions referenced in the issue. If no specific code is referenced or obvious, explicitly output 'None'."
    )
    code: List[str] = Field(
        description="List of exact code snippets, function names, or stack traces found in the issue text. If no code is found, the first element MUST be 'None'."
    )

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------

def setup_logging(repo: str):
    """Configures Logfire to write to a local file and enables LiteLLM debugging."""
    # Turn on LiteLLM's verbose stdout debugging
    litellm._turn_on_debug()
    
    # Hook LiteLLM directly into Logfire
    litellm.success_callback = ["logfire"]
    litellm.failure_callback = ["logfire"]

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Format log name: logs/scverse_scanpy_20240501_143000.txt
    safe_repo = repo.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{safe_repo}_{timestamp}.txt")
    
    log_file = open(log_path, "a", encoding="utf-8")
    
    # Configure Logfire to output OpenTelemetry spans locally
    logfire.configure(
        send_to_logfire=False, # Keep data strictly local
        console=logfire.ConsoleOptions(min_log_level='info'),
        additional_span_processors=[
            SimpleSpanProcessor(ConsoleSpanExporter(out=log_file))
        ]
    )
    print(f"-> Diagnostics enabled. Logs saving to: {os.path.abspath(log_path)}")

def load_environment() -> Dict[str, str]:
    """Loads environment variables securely."""
    if os.path.exists(".my_env"):
        load_dotenv(dotenv_path=".my_env")
    else:
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
# LLM ENGINE
# ---------------------------------------------------------

def digest_issue_text(issue_title: str, issue_body: str, model_name: str) -> IssueDigest:
    """
    Uses an LLM wrapped in Instructor to extract structured RAG metadata from an issue.
    """
    # Wrap LiteLLM with Instructor for native Pydantic support across providers
    client = instructor.from_litellm(litellm.completion)
    
    # Bounded behavior: truncate massive stack traces to save tokens
    safe_body = str(issue_body)[:3000] if issue_body else "No description provided."
    
    system_prompt = (
        "You are a Senior Staff Engineer analyzing a GitHub issue. "
        "Your goal is to extract high-signal search terms, evaluate referenced code, "
        "and isolate stack traces or code snippets. Be highly precise."
    )
    user_prompt = f"Title: {issue_title}\n\nBody: {safe_body}"
    
    try:
        digest = client.chat.completions.create(
            model=model_name,
            response_model=IssueDigest,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=2 # Automatically retry if the LLM hallucinates schema
        )
        return digest
    except Exception as e:
        print(f"[dim red]LLM parsing failed for issue: {e}. Falling back to default.[/dim red]")
        # Graceful fallback if the API fails
        return IssueDigest(
            key_phrases=[issue_title], 
            code_reasoning="None", 
            code=["None"]
        )

# ---------------------------------------------------------
# GITHUB API INTEGRATION
# ---------------------------------------------------------

def fetch_repo_readme(repo: str, headers: Dict[str, str]) -> str:
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
        return [issue for issue in issues if "pull_request" not in issue]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching issues: {e}")
        sys.exit(1)

def fetch_remote_python_chunks(repo: str, headers: Dict[str, str], max_files: int = 200) -> List[Dict[str, str]]:
    print(f"Fetching repository tree for {repo}...")
    try:
        repo_info = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=10).json()
        branch = repo_info.get("default_branch", "main")
        
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        tree_data = requests.get(tree_url, headers=headers, timeout=10).json()
        
        py_files = []
        for item in tree_data.get("tree", []):
            path = item.get("path", "")
            if path.endswith(".py") and "tests/" not in path and "docs/" not in path and not path.startswith("."):
                py_files.append(path)
                
        target_files = py_files[:max_files]
        if not target_files:
            return []
            
        print(f"-> AST parsing {len(target_files)} remote Python files (Max={max_files})...")
        
        chunks = []
        file_headers = headers.copy()
        file_headers["Accept"] = "application/vnd.github.v3.raw"
        
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), transient=True) as progress:
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
                                end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 10
                                chunks.append({
                                    "section_header": f"{path} - {node.name}",
                                    "text": "\n".join(file_lines[start:end]),
                                    "filepath": path
                                })
                except Exception:
                    pass
                progress.advance(task)
        return chunks
    except Exception as e:
        print(f"Warning: Remote codebase indexing failed ({e}).")
        return []

# ---------------------------------------------------------
# LIGHTWEIGHT RAG / SEARCH ENGINE
# ---------------------------------------------------------

def chunk_by_headers(markdown_text: str) -> List[Dict[str, str]]:
    if not markdown_text: return []
    lines = markdown_text.strip().split('\n')
    chunks, current_header, current_content = [], "General Context", []
    
    for line in lines:
        if line.startswith('#'):
            if current_content:
                chunks.append({"section_header": current_header, "text": " ".join(current_content).strip()})
                current_content = []
            current_header = re.sub(r'^#+\s*', '', line).strip()
        elif line.strip():
            current_content.append(line.strip())
            
    if current_content:
        chunks.append({"section_header": current_header, "text": " ".join(current_content).strip()})
    return chunks

def search_repo_docs(query: str, chunks: List[Dict[str, str]], top_k: int = 2) -> List[Dict[str, str]]:
    if not chunks: return []
    query_terms = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split()) | set(chunk["section_header"].lower().split())
        overlap = len(query_terms.intersection(chunk_words))
        scored_chunks.append((overlap, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:top_k]]

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():
    config = load_environment()
    
    parser = argparse.ArgumentParser(description="AI-powered GitHub Issue Triage Agent.")
    parser.add_argument("--repo", type=str, default=config["TARGET_REPO"])
    parser.add_argument("--chunk_size", type=int, default=5)
    args = parser.parse_args()

    if not args.repo:
        print("ERROR: No repository specified. Provide --repo or set TARGET_REPO in your .env")
        sys.exit(1)

    # Initialize logging now that we have the repo name
    setup_logging(args.repo)

    headers = {
        "Authorization": f"token {config['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json"
    }

    print(f"\n--- Initializing Triage Pipeline for {args.repo} ---")
    
    readme_markdown = fetch_repo_readme(args.repo, headers)
    doc_chunks = chunk_by_headers(readme_markdown)
    print(f"-> Parsed README into {len(doc_chunks)} chunks.")
    
    code_chunks = fetch_remote_python_chunks(args.repo, headers, max_files=200)
    print(f"-> Indexed {len(code_chunks)} code chunks.")
    
    page = 1
    while True:
        print(f"\n=== Fetching Batch {page} ===")
        issues = fetch_github_issues(args.repo, headers, limit=args.chunk_size, page=page)
        
        if not issues:
            print("\n[success] No more active issues found. Triage complete![/success]")
            break
            
        for i, sample_issue in enumerate(issues, 1):
            issue_title = sample_issue['title']
            issue_body = sample_issue.get('body', '')
            print(f"\n[Test] Issue #{sample_issue['number']}: {issue_title}")
            
            # 1. Digest the Issue Text
            digest = digest_issue_text(issue_title, issue_body, config["MODEL_NAME"])
            
            print(f"  -> Generated Phrases:  {', '.join(digest.key_phrases)}")
            print(f"  -> Extracted Code:     {digest.code[0] if digest.code else 'None'}")
            print(f"  -> LLM Reasoning:      {digest.code_reasoning[:120]}...")
            
            # 2. Search using LLM-extracted Key Phrases (Query Rewriting)
            search_query = " ".join(digest.key_phrases)
            
            relevant_docs = search_repo_docs(search_query, doc_chunks, top_k=1)
            print(f"  -> Docs Match:         '{relevant_docs[0]['section_header'] if relevant_docs else 'None'}'")
                
            relevant_code = search_repo_docs(search_query, code_chunks, top_k=1)
            print(f"  -> Code Match:         '{relevant_code[0]['section_header'] if relevant_code else 'None'}'")
        
        print("\n" + "-"*50)
        user_input = input(f"Press [Enter] to fetch the next {args.chunk_size} issues, or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            break
        page += 1

if __name__ == "__main__":
    main()