import re
from datetime import datetime
from collections import defaultdict

def cluster_task_data(jira_data, git_data, email_data):
    # Step 1: Create task clusters with JIRA tickets as foundation
    task_clusters = defaultdict(lambda: {"jira": None, "git_commits": [], "emails": [], "timeline": []})
    
    # Step 2: Process JIRA tickets
    for ticket in jira_data:
        task_id = ticket["id"]
        task_clusters[task_id]["jira"] = ticket
        
        # Add to timeline
        task_clusters[task_id]["timeline"].append({
            "type": "jira",
            "id": task_id,
            "timestamp": datetime.fromisoformat(ticket["created_at"]),
            "data": ticket
        })
    
    # Step 3: Process git commits
    for commit in git_data:
        task_id = commit.get("ticket")
        if task_id:
            # Add commit to its task cluster
            task_clusters[task_id]["git_commits"].append(commit)
            
            # Add to timeline
            task_clusters[task_id]["timeline"].append({
                "type": "git",
                "id": commit["commit_id"],
                "timestamp": datetime.fromisoformat(commit["timestamp"]),
                "data": commit
            })
    
    # Step 4: Process emails
    for email in email_data:
        # Extract task IDs from subject and body
        task_ids = set(re.findall(r'PROJ-\d+', email["subject"] + email["body"]))
        
        # Extract commit IDs from body
        commit_ids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', email["body"])
        
        # Map emails to tasks via task IDs
        for task_id in task_ids:
            if task_id in task_clusters:
                task_clusters[task_id]["emails"].append(email)
                
                # Add to timeline
                task_clusters[task_id]["timeline"].append({
                    "type": "email",
                    "id": email["thread_id"],
                    "timestamp": datetime.fromisoformat(email["timestamp"]),
                    "data": email
                })
        
        # Map emails to tasks via commit IDs
        if not task_ids:  # Only if no direct task references
            for commit_id in commit_ids:
                # Find which task this commit belongs to
                for task_id, cluster in task_clusters.items():
                    if any(commit["commit_id"] == commit_id for commit in cluster["git_commits"]):
                        # Add email to this task cluster
                        if email not in cluster["emails"]:
                            cluster["emails"].append(email)
                            
                            # Add to timeline
                            cluster["timeline"].append({
                                "type": "email",
                                "id": email["thread_id"],
                                "timestamp": datetime.fromisoformat(email["timestamp"]),
                                "data": email
                            })
    
    # Step 5: Sort timeline entries chronologically
    for task_id, cluster in task_clusters.items():
        cluster["timeline"].sort(key=lambda x: x["timestamp"])
    
    # Step 6: Create final structured clusters
    final_clusters = []
    for task_id, cluster in task_clusters.items():
        # Skip empty clusters or those without JIRA ticket
        if not cluster["jira"]:
            continue
            
        final_clusters.append({
            "task_id": task_id,
            "summary": cluster["jira"]["summary"],
            "status": cluster["jira"]["status"],
            "assignee": cluster["jira"]["assignee"],
            "created_at": cluster["jira"]["created_at"],
            "git_commits": cluster["git_commits"],
            "emails": cluster["emails"],
            "timeline": cluster["timeline"],
            # Generate text representation for embedding
            # "text_representation": generate_cluster_text(cluster)
        })
    
    return final_clusters
