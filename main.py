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
from typing import List, Dict, Any, Optional, Literal
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
# GLOBAL DEFAULTS
# ---------------------------------------------------------
PAGE_SIZE = 1            # Number of issues to fetch per interactive batch (set to 1 for debugging/compute saving)
MAX_CODE_FILES = 200     # Maximum number of remote Python files to parse via AST

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
        description="Brief reasoning (1-2 sentences max) that primes the agent to extract relevant code chunks or single functions referenced in the issue. If no specific code is referenced or obvious, explicitly output 'None'."
    )
    code: List[str] = Field(
        description="List of exact code snippets, function names, or stack traces found in the issue text. If no code is found, the first element MUST be 'None'."
    )

class RetrievedChunk(BaseModel):
    """Represents a relevant section of code or documentation retrieved via RAG."""
    source_file: str = Field(description="The filepath of the matched document or code.")
    content: str = Field(description="The raw markdown text or Python source code of the chunk.")
    line_number: Optional[int] = Field(default=None, description="The starting line number if it's a code chunk.")

class TriageContext(BaseModel):
    """
    The complete, enriched context payload that will be fed to the final LLM 
    for upstream issue determination and triage decisions.
    """
    issue_title: str
    issue_body: str
    extracted_key_phrases: List[str]
    extracted_code_references: List[str]
    retrieved_chunks: List[RetrievedChunk]

class TriageResult(BaseModel):
    """
    The final triaged output containing actionable decisions for the maintainer.
    Uses strict Literals to prevent LLM hallucination of categories.
    """
    issue_summary: str = Field(
        description="A concise 1-2 sentence summary of the core issue being reported."
    )
    investigation_target: str = Field(
        description="Specific coding lines, function names, or documentation sections to investigate based on the RAG context."
    )
    upstream_risk: Literal["Low", "Medium", "High"] = Field(
        description="Risk that this issue is caused by an external dependency (upstream) rather than the core codebase."
    )
    triage_priority: Literal["Low", "Medium", "High"] = Field(
        description="The priority level for addressing this issue."
    )
    triage_reasoning: str = Field(
        description="Reasoning for the assigned triage priority and upstream risk."
    )
    github_label: Literal["bug", "enhancement", "documentation","invalid"] = Field(
        description="The most appropriate GitHub label for this issue. Issues are invalid if they are requests for usage guidamce"
    )
    label_reasoning: str = Field(
        description="Reasoning for the chosen GitHub label."
    )
    further_info_required: List[Literal["Code", "Error", "More information"]] = Field(
        description="What additional information is needed from the user to reproduce or fix the issue? Leave empty if no further information is needed."
    )

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------

def setup_logging(repo: str):
    """Configures Logfire to write to a local file. Removes noisy LiteLLM stdout."""
    # Suppress ALL noisy internal LiteLLM warnings and stack traces
    litellm.suppress_debug_info = True
    
    # We removed the litellm callbacks that cause the LOGFIRE_TOKEN crash.
    # We rely entirely on our manual logfire.info() calls to keep logs clean.

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Format log name: logs/scverse_scanpy_20240501_143000.txt
    safe_repo = repo.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{safe_repo}_{timestamp}.txt")
    
    log_file = open(log_path, "a", encoding="utf-8")
    
    # Configure Logfire to output OpenTelemetry spans locally
    # We set console formatting to WARNING so our info statements don't clutter the CLI UX
    logfire.configure(
        send_to_logfire=False, # Keep data strictly local
        console=logfire.ConsoleOptions(min_log_level='warning'),
        additional_span_processors=[
            SimpleSpanProcessor(ConsoleSpanExporter(out=log_file))
        ]
    )
    logfire.info("Diagnostics enabled. Logs saving to: {log_path}", log_path=os.path.abspath(log_path))

def load_environment() -> Dict[str, str]:
    """Loads environment variables securely."""
    # Force the .env file to overwrite any old terminal session variables
    if os.path.exists(".env"):
        load_dotenv(dotenv_path=".env", override=True)
    else:
        load_dotenv(override=True)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        # Before logging is initialized, we use standard print
        print("ERROR: GITHUB_TOKEN not found. Please check your .env file.")
        sys.exit(1)
        
    # Aggressively scrub API keys to prevent hidden space/quote errors
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
        val = os.getenv(key_name)
        if val:
            # Strip quotes, newlines, and trailing spaces that cause 401 errors
            clean_val = val.strip(' "\'\n\r')
            os.environ[key_name] = clean_val
            
            if key_name == "ANTHROPIC_API_KEY":
                logfire.info(f"Loaded & Scrubbed Anthropic Key: {clean_val[:10]}... (Length: {len(clean_val)})")

    return {
        "GITHUB_TOKEN": token,
        "TARGET_REPO": os.getenv("TARGET_REPO"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "claude-3-5-haiku-20241022"), # Pinned for reproducibility
        "GITHUB_API_URL": "https://api.github.com"
    }

# ---------------------------------------------------------
# LLM ENGINES
# ---------------------------------------------------------

def digest_issue_text(issue_title: str, issue_body: str, model_name: str) -> IssueDigest:
    """
    Step 1 LLM Call: Uses an LLM to extract structured RAG metadata from a raw issue.
    """
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
            max_retries=2
        )
        return digest
    except Exception as e:
        error_msg = str(e)
        logfire.error("LLM parsing failed for issue. Error details: {error}", error=error_msg)
        
        if "not_found_error" in error_msg and "model" in error_msg:
            print("\n" + "!"*60)
            print("🚨 MODEL NOT FOUND ERROR 🚨")
            print(f"The API rejected the model name: '{model_name}'")
            print("Anthropic often deprecates older model versions. Please update your .env file.")
            print("!"*60 + "\n")
            
        elif "invalid x-api-key" in error_msg or "authentication_error" in error_msg:
            print("\n" + "!"*60)
            print("🚨 ANTHROPIC AUTHENTICATION ERROR 🚨")
            print("Anthropic actively rejected your key.")
            print("!"*60 + "\n")
            
        return IssueDigest(key_phrases=[issue_title], code_reasoning="None", code=["None"])

def generate_triage_decision(context: TriageContext, model_name: str) -> TriageResult:
    """
    Step 2 LLM Call: Takes the fully enriched TriageContext and performs the 
    final LLM call to determine issue severity, upstream dependencies, and draft replies.
    """
    client = instructor.from_litellm(litellm.completion)
    
    system_prompt = (
        "You are a Senior Open Source Maintainer triaging incoming GitHub issues. "
        "You have been provided with an enriched context payload containing the original issue, "
        "extracted signals, and code/documentation snippets retrieved from the repository codebase. "
        "Synthesize this information to provide a highly accurate, structured triage decision."
    )
    
    # Dump the Pydantic schema to JSON so the LLM gets a perfectly formatted payload
    user_prompt = f"=== ENRICHED TRIAGE CONTEXT ===\n{context.model_dump_json(indent=2)}"
    
    logfire.info("-> Executing final triage decision with Enriched Context.")
    
    try:
        decision = client.chat.completions.create(
            model=model_name,
            response_model=TriageResult,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=2
        )
        return decision
    except Exception as e:
        logfire.error("LLM final triage failed. Error details: {error}", error=str(e))
        return TriageResult(
            issue_summary="Triage failed due to API error.",
            investigation_target="Unknown",
            upstream_risk="Low",
            triage_priority="Low",
            triage_reasoning=f"API Error fallback: {str(e)}",
            github_label="bug",
            label_reasoning="API Error fallback.",
            further_info_required=[]
        )

# ---------------------------------------------------------
# GITHUB API INTEGRATION
# ---------------------------------------------------------

def fetch_repo_readme(repo: str, headers: Dict[str, str]) -> str:
    url = f"https://api.github.com/repos/{repo}/readme"
    readme_headers = headers.copy()
    readme_headers["Accept"] = "application/vnd.github.v3.raw"
    
    logfire.info("Fetching README for {repo}...", repo=repo)
    try:
        response = requests.get(url, headers=readme_headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logfire.warn("Could not fetch README. Details: {e}", e=str(e))
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
    
    logfire.info("Fetching {limit} issues from {repo} (Page {page})...", limit=limit, repo=repo, page=page)
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        issues = response.json()
        return [issue for issue in issues if "pull_request" not in issue]
    except requests.exceptions.RequestException as e:
        logfire.error("Error fetching issues: {e}", e=str(e))
        sys.exit(1)

def fetch_remote_python_chunks(repo: str, headers: Dict[str, str], max_files: int = 200) -> List[Dict[str, str]]:
    logfire.info("Fetching repository tree for {repo}...", repo=repo)
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
            
        logfire.info("AST parsing {count} remote Python files (Max={max_files})...", count=len(target_files), max_files=max_files)
        
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
                                    "filepath": path,
                                    "line_number": node.lineno
                                })
                except Exception:
                    pass
                progress.advance(task)
        return chunks
    except Exception as e:
        logfire.warn("Remote codebase indexing failed ({e}).", e=str(e))
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
                chunks.append({
                    "section_header": current_header, 
                    "text": " ".join(current_content).strip(),
                    "filepath": "README.md",
                    "line_number": None
                })
                current_content = []
            current_header = re.sub(r'^#+\s*', '', line).strip()
        elif line.strip():
            current_content.append(line.strip())
            
    if current_content:
        chunks.append({
            "section_header": current_header, 
            "text": " ".join(current_content).strip(),
            "filepath": "README.md",
            "line_number": None
        })
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
    parser.add_argument("--page_size", type=int, default=PAGE_SIZE, help="Number of issues to fetch per batch")
    parser.add_argument("--max_code_files", type=int, default=MAX_CODE_FILES, help="Maximum number of remote Python files to index")
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

    # Use standard prints for clean CLI UX; logfire records the same data invisibly
    print(f"\n--- Initializing Triage Pipeline for {args.repo} ---")
    logfire.info("--- Initializing Triage Pipeline for {repo} ---", repo=args.repo)
    
    # 1. Fetch Documentation
    readme_markdown = fetch_repo_readme(args.repo, headers)
    doc_chunks = chunk_by_headers(readme_markdown)
    print(f"-> Parsed README into {len(doc_chunks)} distinct documentation chunks.")
    logfire.info("Parsed README into {count} chunks.", count=len(doc_chunks))
    
    # 2. Fetch Codebase
    code_chunks = fetch_remote_python_chunks(args.repo, headers, max_files=args.max_code_files)
    print(f"-> Indexed {len(code_chunks)} code chunks from the Abstract Syntax Tree (AST).")
    logfire.info("Indexed {count} code chunks.", count=len(code_chunks))
    
    # Combine everything into a unified search pool
    all_chunks = doc_chunks + code_chunks
    
    page = 1
    while True:
        print(f"\n=== Fetching Batch {page} ===")
        logfire.info("=== Fetching Batch {page} ===", page=page)
        
        issues = fetch_github_issues(args.repo, headers, limit=args.page_size, page=page)
        
        if not issues:
            print("\n[success] No more active issues found. Triage complete![/success]")
            logfire.info("No more active issues found. Triage complete!")
            break
            
        for i, sample_issue in enumerate(issues, 1):
            issue_title = sample_issue['title']
            issue_body = sample_issue.get('body', '')
            
            print(f"\n[Issue #{sample_issue['number']}] {issue_title}")
            logfire.info("[Test] Issue #{number}: {title}", number=sample_issue['number'], title=issue_title)
            
            # --- PHASE 1: Digest the Issue Text ---
            digest = digest_issue_text(issue_title, issue_body, config["MODEL_NAME"])
            logfire.info("Generated Phrases: {phrases}", phrases=', '.join(digest.key_phrases))
            
            # --- PHASE 2: Enriched RAG Search (Docs + Code) ---
            valid_code_terms = [c for c in digest.code if c.lower() != "none"]
            combined_terms = digest.key_phrases + valid_code_terms
            search_query = " ".join(combined_terms)
            
            logfire.info("Enriched Search Query: '{query}'", query=search_query)
            best_matches = search_repo_docs(search_query, all_chunks, top_k=2)
            
            # Format matches for context building
            retrieved_chunks_payload = []
            for match in best_matches:
                retrieved_chunks_payload.append(RetrievedChunk(
                    source_file=match.get('filepath', 'Unknown'),
                    content=match['text'],
                    line_number=match.get('line_number')
                ))
            
            # --- PHASE 3: Final Triage Decision ---
            triage_context = TriageContext(
                issue_title=issue_title,
                issue_body=issue_body,
                extracted_key_phrases=digest.key_phrases,
                extracted_code_references=digest.code,
                retrieved_chunks=retrieved_chunks_payload
            )
            
            decision = generate_triage_decision(triage_context, config["MODEL_NAME"])
            
            print("\n  [FINAL TRIAGE REPORT]")
            print(f"  -> Summary:            {decision.issue_summary}")
            print(f"  -> Investigate:        {decision.investigation_target}")
            print(f"  -> Upstream Risk:      {decision.upstream_risk}")
            print(f"  -> Priority:           {decision.triage_priority}")
            print(f"  -> Priority Reasoning: {decision.triage_reasoning}")
            print(f"  -> GitHub Label:       '{decision.github_label}' ({decision.label_reasoning})")
            
            missing_info_str = ", ".join(decision.further_info_required) if decision.further_info_required else "None"
            print(f"  -> Missing Info:       {missing_info_str}")
            
        print("\n" + "-"*50)
        user_input = input(f"Press [Enter] to fetch the next {args.page_size} issues, or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            break
        page += 1

if __name__ == "__main__":
    main()