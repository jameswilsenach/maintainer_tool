"""
JSON Query Engine for Triage Reports
Acts as a local NoSQL query tool to filter the generated triage JSON 
and produce specialized Markdown sub-reports (e.g., High Priority only).
"""

import os
import glob
import json
import argparse
import time
from typing import List, Dict, Any

def get_latest_json_report(repo: str = None) -> str:
    """Finds the most recently created JSON report in the reports directory."""
    os.makedirs("reports", exist_ok=True)
    
    # If a repo is provided, filter by it, otherwise grab any json report
    search_prefix = repo.replace("/", "_") if repo else "*"
    search_pattern = os.path.join("reports", f"{search_prefix}_triage_*.json")
    
    files = glob.glob(search_pattern)
    if not files:
        return None
        
    # Return the file with the most recent modification time
    return max(files, key=os.path.getmtime)

def generate_markdown_subreport(issues: List[Dict[str, Any]], title: str, output_path: str):
    """Generates a clean markdown report from a filtered list of JSON issue records."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Issues:** {len(issues)}\n\n")
        f.write("---\n\n")

        if not issues:
            f.write("*No issues matched this query criteria.*\n")
            return

        for record in issues:
            issue_num = record.get("issue_number", "Unknown")
            issue_title = record.get("issue_title", "Untitled")
            issue_url = record.get("issue_url", f"https://github.com/issues/{issue_num}")
            
            # The structured LLM output lives inside the 'triage_decision' key
            decision = record.get("triage_decision", {})
            
            f.write(f"## [Issue #{issue_num}]({issue_url}): {issue_title}\n\n")
            f.write(f"**Issue Summary:** {decision.get('issue_summary', 'N/A')}\n\n")
            f.write(f"- **Investigate Target:** {decision.get('investigation_target', 'N/A')}\n")
            
            f.write(f"- **GitHub Label:** `{decision.get('github_label', 'N/A')}` (Reasoning: {decision.get('label_reasoning', 'N/A')})\n")
            
            missing_info = decision.get("further_info_required", [])
            missing_info_str = ", ".join(missing_info) if missing_info else "None"
            f.write(f"- **Further Info Required:** {missing_info_str}\n")
            
            f.write(f"- **Upstream Risk:** {decision.get('upstream_risk', 'N/A')}\n")
            f.write(f"- **Triage Priority:** {decision.get('triage_priority', 'N/A')} (Reasoning: {decision.get('triage_reasoning', 'N/A')})\n\n")
            
            f.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description="Query the Triage JSON NoSQL Database to generate sub-reports.")
    parser.add_argument("--input_json", type=str, default=None, help="Specific JSON report to parse. Defaults to the most recent.")
    parser.add_argument("--repo", type=str, default=None, help="Repository name to help find the latest JSON if --input_json is not provided.")
    
    # Filter Flags
    parser.add_argument("--high_priority", action="store_true", help="Generate a report of only High priority issues.")
    parser.add_argument("--low_priority", action="store_true", help="Generate a report of only Low priority issues.")
    parser.add_argument("--high_upstream", action="store_true", help="Generate a report of issues with High upstream risk.")
    parser.add_argument("--needs_info", action="store_true", help="Generate a report of issues requiring more information from the user.")
    args = parser.parse_args()

    # 1. Resolve the target JSON file
    target_file = args.input_json
    if not target_file:
        target_file = get_latest_json_report(args.repo)
        
    if not target_file or not os.path.exists(target_file):
        print(f"ERROR: Could not find a valid JSON report to parse.")
        sys.exit(1)

    print(f"-> Loading NoSQL data from: {target_file}")
    
    with open(target_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: {target_file} is not a valid JSON file.")
            sys.exit(1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(target_file))[0]
    
    # Track if we actually ran any queries
    queries_run = False

    # 2. Execute Queries and Generate Sub-Reports
    
    if args.high_priority:
        queries_run = True
        filtered_issues = [issue for issue in data if issue.get("triage_decision", {}).get("triage_priority") == "High"]
        out_path = os.path.join("reports", f"{base_filename}_HIGH_PRIORITY_{timestamp}.md")
        generate_markdown_subreport(filtered_issues, "🚨 Action Required: High Priority Issues", out_path)
        print(f"[Query Success] Found {len(filtered_issues)} High Priority issues -> {out_path}")

    if args.low_priority:
        queries_run = True
        filtered_issues = [issue for issue in data if issue.get("triage_decision", {}).get("triage_priority") == "Low"]
        out_path = os.path.join("reports", f"{base_filename}_LOW_PRIORITY_{timestamp}.md")
        generate_markdown_subreport(filtered_issues, "🧊 Backlog: Low Priority Issues", out_path)
        print(f"[Query Success] Found {len(filtered_issues)} Low Priority issues -> {out_path}")

    if args.high_upstream:
        queries_run = True
        filtered_issues = [issue for issue in data if issue.get("triage_decision", {}).get("upstream_risk") == "High"]
        out_path = os.path.join("reports", f"{base_filename}_UPSTREAM_RISK_{timestamp}.md")
        generate_markdown_subreport(filtered_issues, "🔗 External Dependencies: High Upstream Risk", out_path)
        print(f"[Query Success] Found {len(filtered_issues)} High Upstream Risk issues -> {out_path}")
        
    if args.needs_info:
        queries_run = True
        # Filter issues where further_info_required exists and does NOT solely contain "None"
        filtered_issues = [
            issue for issue in data 
            if issue.get("triage_decision", {}).get("further_info_required") 
            and "None" not in issue.get("triage_decision", {}).get("further_info_required", [])
        ]
        out_path = os.path.join("reports", f"{base_filename}_NEEDS_INFO_{timestamp}.md")
        generate_markdown_subreport(filtered_issues, "❓ Blocked: Needs User Information", out_path)
        print(f"[Query Success] Found {len(filtered_issues)} issues needing more info -> {out_path}")

    if not queries_run:
        print("No query flags provided. Try running with --high_priority, --high_upstream, etc.")

if __name__ == "__main__":
    main()