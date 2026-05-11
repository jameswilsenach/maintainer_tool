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
import json
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
PAGE_SIZE = 5            # Number of issues to fetch per interactive batch (set to 1 for debugging/compute saving)
MAX_PAGE_SIZE = 10       # Hard cap on batch size to prevent accidental API credit burn
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
    label_reasoning: str = Field(
        description="Reasoning for the chosen GitHub label. Issues are invalid if they are requests for usage guidance."
    )
    github_label: Literal["bug", "enhancement", "documentation", "invalid"] = Field(
        description="The most appropriate GitHub label for this issue."
    )
    further_info_required: List[Literal["Code", "Error", "More information", "None"]] = Field(
        description="What additional information is needed from the user to reproduce or fix the issue? None if no further information is needed."
    )
    triage_reasoning: str = Field(
        description="Reasoning for the assigned triage priority and upstream risk."
    )
    upstream_risk: Literal["Low", "Medium", "High"] = Field(
        description="Risk that this issue is caused by an external dependency (upstream) rather than the core codebase."
    )
    triage_priority: Literal["Low", "Medium", "High"] = Field(
        description="The priority level for addressing this issue."
    )

# ---------------------------------------------------------
# SETUP & CONFIGURATION
# ---------------------------------------------------------

def setup_logging(repo: str):
    """Configures Logfire to write to a local file. Removes noisy LiteLLM stdout."""
    litellm.suppress_debug_info = True

    os.makedirs("logs", exist_ok=True)

    safe_repo = repo.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", f"{safe_repo}_{timestamp}.txt")
    
    log_file = open(log_path, "a", encoding="utf-8")
    
    logfire.configure(
        send_to_logfire=False, 
        console=logfire.ConsoleOptions(min_log_level='warning'),
        additional_span_processors=[
            SimpleSpanProcessor(ConsoleSpanExporter(out=log_file))
        ]
    )
    logfire.info("Diagnostics enabled. Logs saving to: {log_path}", log_path=os.path.abspath(log_path))

def load_environment() -> Dict[str, str]:
    if os.path.exists(".env"):
        load_dotenv(dotenv_path=".env", override=True)
    else:
        load_dotenv(override=True)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERROR: GITHUB_TOKEN not found. Please check your .env file.")
        sys.exit(1)
        
    for key_name in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]:
        val = os.getenv(key_name)
        if val:
            clean_val = val.strip(' "\'\n\r')
            os.environ[key_name] = clean_val
            if key_name == "ANTHROPIC_API_KEY":
                logfire.info(f"Loaded & Scrubbed Anthropic Key: {clean_val[:10]}... (Length: {len(clean_val)})")

    return {
        "GITHUB_TOKEN": token,
        "TARGET_REPO": os.getenv("TARGET_REPO"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "claude-3-5-haiku-20241022"), 
        "GITHUB_API_URL": "https://api.github.com"
    }

# ---------------------------------------------------------
# LLM ENGINES
# ---------------------------------------------------------

def digest_issue_text(issue_title: str, issue_body: str, model_name: str) -> IssueDigest:
    client = instructor.from_litellm(litellm.completion)
    safe_body = str(issue_body)[:3000] if issue_body else "No description provided."
    
    system_prompt = (
        "You are a Senior Staff Engineer analyzing a GitHub issue. "
        "Your goal is to extract high-signal search terms, evaluate referenced code, "
        "and isolate stack traces or code snippets. Be highly precise."
    )
    user_prompt = f"Title: {issue_title}\n\nBody: {safe_body}"
    
    try:
        return client.chat.completions.create(
            model=model_name,
            response_model=IssueDigest,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=2
        )
    except Exception as e:
        error_msg = str(e)
        logfire.error("LLM parsing failed for issue. Error details: {error}", error=error_msg)
        
        if "not_found_error" in error_msg and "model" in error_msg:
            print("\n" + "!"*60)
            print("MODEL NOT FOUND ERROR")
            print(f"The API rejected the model name: '{model_name}'")
            print("Anthropic often deprecates older model versions. Please update your .env file.")
            print("!"*60 + "\n")
        elif "invalid x-api-key" in error_msg or "authentication_error" in error_msg:
            print("\n" + "!"*60)
            print("ANTHROPIC AUTHENTICATION ERROR")
            print("Anthropic actively rejected your key.")
            print("!"*60 + "\n")
            
        return IssueDigest(key_phrases=[issue_title], code_reasoning="None", code=["None"])

def generate_triage_decision(context: TriageContext, model_name: str) -> TriageResult:
    client = instructor.from_litellm(litellm.completion)
    
    system_prompt = (
        "You are a Senior Open Source Maintainer triaging incoming GitHub issues. "
        "You have been provided with an enriched context payload containing the original issue, "
        "extracted signals, and code/documentation snippets retrieved from the repository codebase. "
        "Synthesize this information to provide a highly accurate, structured triage decision."
    )
    
    user_prompt = f"=== ENRICHED TRIAGE CONTEXT ===\n{context.model_dump_json(indent=2)}"
    logfire.info("-> Executing final triage decision with Enriched Context.")
    
    try:
        return client.chat.completions.create(
            model=model_name,
            response_model=TriageResult,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=2
        )
    except Exception as e:
        logfire.error("LLM final triage failed. Error details: {error}", error=str(e))
        return TriageResult(
            issue_summary="Triage failed due to API error.",
            investigation_target="Unknown",
            label_reasoning="API Error fallback.",
            github_label="bug",
            further_info_required=["None"],
            triage_reasoning=f"API Error fallback: {str(e)}",
            upstream_risk="Low",
            triage_priority="Low"
        )

# ---------------------------------------------------------
# GITHUB API INTEGRATION
# ---------------------------------------------------------

def fetch_repo_readme(repo: str, headers: Dict[str, str]) -> str:
    url = f"https://api.github.com/repos/{repo}/readme"
    readme_headers = headers.copy()
    readme_headers["Accept"] = "application/vnd.github.v3.raw"
    try:
        response = requests.get(url, headers=readme_headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        return ""

def fetch_github_issues(repo: str, headers: Dict[str, str], limit: int = 30, page: int = 1) -> List[Dict[str, Any]]:
    """Fetches a larger batch of issues to populate the processing buffer."""
    url = f"https://api.github.com/repos/{repo}/issues"
    params = {"state": "open", "sort": "created", "direction": "desc", "per_page": limit, "page": page}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        issues = response.json()
        return [issue for issue in issues if "pull_request" not in issue]
    except requests.exceptions.RequestException as e:
        logfire.error("Error fetching issues: {e}", e=str(e))
        sys.exit(1)

def fetch_remote_python_chunks(repo: str, headers: Dict[str, str], max_files: int = 200) -> List[Dict[str, str]]:
    try:
        repo_info = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=10).json()
        branch = repo_info.get("default_branch", "main")
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        tree_data = requests.get(tree_url, headers=headers, timeout=10).json()
        
        py_files = [item.get("path") for item in tree_data.get("tree", []) 
                    if item.get("path", "").endswith(".py") and "tests/" not in item.get("path") 
                    and "docs/" not in item.get("path") and not item.get("path", "").startswith(".")]
                
        target_files = py_files[:max_files]
        if not target_files: return []
        
        chunks = []
        stats = {
            "with_nodes": 0,
            "empty_or_no_nodes": 0,
            "failed_network": 0,
            "failed_ast": 0
        }
        node_counts = {"ClassDef": 0, "FunctionDef": 0, "AsyncFunctionDef": 0}
        
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
                        try:
                            module = ast.parse(content)
                            file_lines = content.split('\n')
                            nodes_found = 0
                            
                            for node in module.body:
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                                    # Track exactly what type of chunk we are extracting
                                    if isinstance(node, ast.ClassDef): node_counts["ClassDef"] += 1
                                    elif isinstance(node, ast.FunctionDef): node_counts["FunctionDef"] += 1
                                    elif isinstance(node, ast.AsyncFunctionDef): node_counts["AsyncFunctionDef"] += 1
                                    
                                    start = node.lineno - 1
                                    end = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno else start + 10
                                    chunks.append({
                                        "section_header": f"{path} - {node.name}",
                                        "text": "\n".join(file_lines[start:end]),
                                        "filepath": path,
                                        "line_number": node.lineno
                                    })
                                    nodes_found += 1
                                    
                            if nodes_found > 0:
                                stats["with_nodes"] += 1
                            else:
                                stats["empty_or_no_nodes"] += 1
                                
                        except Exception:
                            stats["failed_ast"] += 1
                    else:
                        stats["failed_network"] += 1
                except Exception:
                    stats["failed_network"] += 1
                progress.advance(task)
                
        # Emit a comprehensive observability metric for the AST phase
        summary_msg = (
            f"AST Extraction: {stats['with_nodes']} files yielded chunks, "
            f"{stats['empty_or_no_nodes']} had no classes/functions. "
            f"[Failures: {stats['failed_network']} Network/API, {stats['failed_ast']} Syntax]"
        )
        
        logfire.info(
            "AST Extraction Summary: {stats} | Node Counts: {node_counts}",
            stats=stats,
            node_counts=node_counts
        )
        print(f"-> {summary_msg}")
        return chunks
    except Exception:
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
                chunks.append({"section_header": current_header, "text": " ".join(current_content).strip(), "filepath": "README.md", "line_number": None})
                current_content = []
            current_header = re.sub(r'^#+\s*', '', line).strip()
        elif line.strip():
            current_content.append(line.strip())
            
    if current_content:
        chunks.append({"section_header": current_header, "text": " ".join(current_content).strip(), "filepath": "README.md", "line_number": None})
    return chunks

def search_repo_docs(query: str, chunks: List[Dict[str, str]], top_k: int = 2) -> List[Dict[str, str]]:
    if not chunks: return []
    query_terms = set(query.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk["text"].lower().split()) | set(chunk["section_header"].lower().split())
        overlap = len(query_terms.intersection(chunk_words))
        
        # Only keep chunks that actually have matching terms!
        if overlap > 0:
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
    parser.add_argument("--page_size", type=int, default=PAGE_SIZE, help="Number of valid issues to process per interactive batch")
    parser.add_argument("--max_code_files", type=int, default=MAX_CODE_FILES, help="Maximum number of remote Python files to index")
    parser.add_argument("--start_issue", type=int, default=None, help="Optional starting issue number. Newer issues will be skipped.")
    parser.add_argument("--include_labeled", action="store_true", help="If set, includes issues that already have labels (by default, labeled issues are skipped).")
    args = parser.parse_args()

    # Safeguard to prevent accidental API credit burn
    if args.page_size > MAX_PAGE_SIZE:
        print(f"⚠️  Warning: --page_size {args.page_size} exceeds the maximum limit.")
        print(f"   Capping batch size at {MAX_PAGE_SIZE} to conserve API credits.")
        args.page_size = MAX_PAGE_SIZE

    if not args.repo:
        print("ERROR: No repository specified. Provide --repo or set TARGET_REPO in your .env")
        sys.exit(1)

    setup_logging(args.repo)

    # Setup the structured markdown and json reports
    os.makedirs("reports", exist_ok=True)
    safe_repo = args.repo.replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join("reports", f"{safe_repo}_triage_{timestamp}.md")
    report_json_path = os.path.join("reports", f"{safe_repo}_triage_{timestamp}.json")
    json_report_data = []
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# GitHub Issue Triage Report\n")
        f.write(f"**Repository:** `{args.repo}`\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

    headers = {
        "Authorization": f"token {config['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json"
    }

    print(f"\n--- Initializing Triage Pipeline for {args.repo} ---")
    print(f"-> Active reports will be saved to:")
    print(f"   - Markdown: {os.path.abspath(report_path)}")
    print(f"   - JSON:     {os.path.abspath(report_json_path)}\n")
    
    readme_markdown = fetch_repo_readme(args.repo, headers)
    doc_chunks = chunk_by_headers(readme_markdown)
    print(f"-> Parsed README into {len(doc_chunks)} distinct documentation chunks.")
    
    code_chunks = fetch_remote_python_chunks(args.repo, headers, max_files=args.max_code_files)
    print(f"-> Indexed {len(code_chunks)} code chunks from the Abstract Syntax Tree (AST).")
    
    all_chunks = doc_chunks + code_chunks
    
    api_page = 1
    issue_buffer = []
    
    while True:
        # Fill the buffer until it has enough valid issues for the next batch display
        while len(issue_buffer) < args.page_size:
            issues = fetch_github_issues(args.repo, headers, limit=30, page=api_page)
            if not issues:
                break
                
            for issue in issues:
                # Filter out issues newer than start_issue if provided
                if args.start_issue and issue['number'] > args.start_issue:
                    logfire.info("Skipping Issue #{num} (Newer than start_issue {start})", num=issue['number'], start=args.start_issue)
                    continue
                
                # Filter out issues that already have labels (assume already triaged)
                if not args.include_labeled and issue.get('labels'):
                    logfire.info("Skipping Issue #{num} (Already labeled/triaged)", num=issue['number'])
                    continue
                    
                issue_buffer.append(issue)
                
            api_page += 1
            
        if not issue_buffer:
            print("\n[SUCCESS] No more active issues found matching criteria. Triage complete!")
            break
            
        # Process the exact page size from the buffer
        batch_to_process = issue_buffer[:args.page_size]
        issue_buffer = issue_buffer[args.page_size:]
            
        for sample_issue in batch_to_process:
            issue_title = sample_issue['title']
            issue_body = sample_issue.get('body', '')
            
            print(f"\n[Issue #{sample_issue['number']}] {issue_title}")
            
            # --- PHASE 1: Digest the Issue Text ---
            digest = digest_issue_text(issue_title, issue_body, config["MODEL_NAME"])
            
            # --- PHASE 2: Enriched RAG Search (Docs + Code) ---
            valid_code_terms = [c for c in digest.code if c.lower() != "none"]
            combined_terms = digest.key_phrases + valid_code_terms
            search_query = " ".join(combined_terms)
            
            best_matches = search_repo_docs(search_query, all_chunks, top_k=2)
            retrieved_chunks_payload = []
            
            # 1. ALWAYS inject the first README chunk to ground the LLM in the repo's purpose
            if doc_chunks:
                retrieved_chunks_payload.append(RetrievedChunk(
                    source_file="README.md (Repository Overview)",
                    content=doc_chunks[0]['text'],
                    line_number=None
                ))
                
            # 2. Inject the semantic search matches
            for match in best_matches:
                # Prevent duplication if the search naturally pulled the overview chunk
                if doc_chunks and match['text'] == doc_chunks[0]['text']:
                    continue
                    
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
            
            # Format Missing Info Securely
            missing_info_list = [info for info in decision.further_info_required if info.lower() != "none"]
            missing_info_str = ", ".join(missing_info_list) if missing_info_list else "None"

            # 1. Print formatted output to the Console
            print("\n  [FINAL TRIAGE REPORT]")
            print(f"  -> Summary:            {decision.issue_summary}")
            print(f"  -> Investigate:        {decision.investigation_target}")
            print(f"  -> GitHub Label:       '{decision.github_label}' ({decision.label_reasoning})")
            print(f"  -> Missing Info:       {missing_info_str}")
            print(f"  -> Upstream Risk:      {decision.upstream_risk}")
            print(f"  -> Priority:           {decision.triage_priority}")
            print(f"  -> Priority Reasoning: {decision.triage_reasoning}")

            # 2. Append cleanly formatted markdown to the report file
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(f"## Issue #{sample_issue['number']}: {issue_title}\n\n")
                f.write(f"**Issue Summary:** {decision.issue_summary}\n\n")
                f.write(f"- **Investigate Target:** {decision.investigation_target}\n")
                
                # Render reasoning without italics to prevent inner markdown collision from the LLM
                f.write(f"- **GitHub Label:** `{decision.github_label}` (Reasoning: {decision.label_reasoning})\n")
                f.write(f"- **Further Info Required:** {missing_info_str}\n")
                f.write(f"- **Upstream Risk:** {decision.upstream_risk}\n")
                f.write(f"- **Triage Priority:** {decision.triage_priority} (Reasoning: {decision.triage_reasoning})\n\n")
                
                # Optionally add the key RAG references that led to this decision
                if retrieved_chunks_payload:
                    f.write("**Key Context Retrieved:**\n")
                    for chunk in retrieved_chunks_payload:
                        line_ref = f" (Line {chunk.line_number})" if chunk.line_number else ""
                        f.write(f"- `{chunk.source_file}`{line_ref}\n")
                
                f.write("\n---\n\n")
                
            # 3. Append to JSON data and rewrite file iteratively
            issue_record = {
                "issue_number": sample_issue['number'],
                "issue_title": issue_title,
                "issue_url": sample_issue.get('html_url', ''),
                "digest": digest.model_dump(),
                "triage_decision": decision.model_dump(),
                "retrieved_context": [chunk.model_dump() for chunk in retrieved_chunks_payload]
            }
            json_report_data.append(issue_record)

            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump(json_report_data, f, indent=2)
            
        print("\n" + "="*50)
        user_input = input(f"Press [Enter] to fetch the next {args.page_size} valid issues, or type 'exit' to quit: ").strip().lower()
        if user_input == 'exit':
            print(f"\nReports completed and safely stored at:")
            print(f"  - Markdown: {os.path.abspath(report_path)}")
            print(f"  - JSON:     {os.path.abspath(report_json_path)}")
            break

if __name__ == "__main__":
    main()