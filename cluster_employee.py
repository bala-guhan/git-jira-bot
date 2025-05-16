import re
from datetime import datetime
from collections import defaultdict

def cluster_employee_data(jira_data, git_data, email_data):
    """
    Clusters data based on employees instead of tasks.
    Creates a comprehensive profile for each employee showing all their activities.
    """
    # Step 1: Create employee clusters
    employee_clusters = defaultdict(lambda: {
        "jira_assigned": [],
        "jira_resolved": [],
        "git_commits": [],
        "emails_sent": [],
        "emails_received": [],
        "timeline": []
    })
    
    # Step 2: Process JIRA tickets
    for ticket in jira_data:
        # Handle assignee data
        assignee = ticket.get("assignee")
        if assignee:
            employee_clusters[assignee]["jira_assigned"].append(ticket)
            
            # Add to timeline
            employee_clusters[assignee]["timeline"].append({
                "type": "jira_assigned",
                "id": ticket["id"],
                "timestamp": datetime.fromisoformat(ticket["created_at"]),
                "data": ticket
            })
        
        # Handle resolver data (if available)
        resolution = ticket.get("resolution", "")
        if resolution:
            # Try to extract resolver from resolution text
            resolver_match = re.search(r"Resolved by (\w+)", resolution)
            if resolver_match:
                resolver = resolver_match.group(1)
                employee_clusters[resolver]["jira_resolved"].append(ticket)
                
                # Add to timeline
                employee_clusters[resolver]["timeline"].append({
                    "type": "jira_resolved",
                    "id": ticket["id"],
                    "timestamp": datetime.fromisoformat(ticket["updated_at"]),
                    "data": ticket
                })
    
    # Step 3: Process git commits
    for commit in git_data:
        author = commit.get("author")
        if author:
            employee_clusters[author]["git_commits"].append(commit)
            
            # Add to timeline
            employee_clusters[author]["timeline"].append({
                "type": "git_commit",
                "id": commit["commit_id"],
                "timestamp": datetime.fromisoformat(commit["timestamp"]),
                "data": commit
            })
    
    # Step 4: Process emails
    for email in email_data:
        sender = email.get("sender")
        if sender:
            employee_clusters[sender]["emails_sent"].append(email)
            
            # Add to timeline
            employee_clusters[sender]["timeline"].append({
                "type": "email_sent",
                "id": email["thread_id"],
                "timestamp": datetime.fromisoformat(email["timestamp"]),
                "data": email
            })
        
        # Process recipients (if available)
        # Note: This requires email data to have a "recipients" field
        recipients = email.get("recipients", [])
        for recipient in recipients:
            employee_clusters[recipient]["emails_received"].append(email)
            
            # Add to timeline
            employee_clusters[recipient]["timeline"].append({
                "type": "email_received",
                "id": email["thread_id"],
                "timestamp": datetime.fromisoformat(email["timestamp"]),
                "data": email
            })
    
    # Step 5: Sort timeline entries chronologically
    for employee, cluster in employee_clusters.items():
        cluster["timeline"].sort(key=lambda x: x["timestamp"])
    
    # Step 6: Create final structured clusters
    final_clusters = []
    for employee, cluster in employee_clusters.items():
        # Skip employees with no activity
        if not (cluster["jira_assigned"] or cluster["jira_resolved"] or 
                cluster["git_commits"] or cluster["emails_sent"] or 
                cluster["emails_received"]):
            continue
        
        # Calculate activity metrics
        total_jira = len(cluster["jira_assigned"]) + len(cluster["jira_resolved"])
        total_commits = len(cluster["git_commits"])
        total_emails = len(cluster["emails_sent"]) + len(cluster["emails_received"])
        
        final_clusters.append({
            "employee": employee,
            "activity_summary": {
                "jira_tickets_assigned": len(cluster["jira_assigned"]),
                "jira_tickets_resolved": len(cluster["jira_resolved"]),
                "git_commits": total_commits,
                "emails_sent": len(cluster["emails_sent"]),
                "emails_received": len(cluster["emails_received"]),
                "total_activity_count": total_jira + total_commits + total_emails
            },
            "jira_assigned": cluster["jira_assigned"],
            "jira_resolved": cluster["jira_resolved"],
            "git_commits": cluster["git_commits"],
            "emails_sent": cluster["emails_sent"],
            "emails_received": cluster["emails_received"],
            "timeline": cluster["timeline"]
        })
    
    # Sort clusters by total activity (most active first)
    final_clusters.sort(key=lambda x: x["activity_summary"]["total_activity_count"], reverse=True)
    
    return final_clusters

def analyze_employee_clusters(employee_clusters):
    """
    Analyzes employee clusters to provide insights about team dynamics.
    """
    total_employees = len(employee_clusters)
    
    # Basic statistics
    stats = {
        "total_employees": total_employees,
        "total_jira_tickets_assigned": sum(e["activity_summary"]["jira_tickets_assigned"] for e in employee_clusters),
        "total_jira_tickets_resolved": sum(e["activity_summary"]["jira_tickets_resolved"] for e in employee_clusters),
        "total_git_commits": sum(e["activity_summary"]["git_commits"] for e in employee_clusters),
        "total_emails_sent": sum(e["activity_summary"]["emails_sent"] for e in employee_clusters),
        "total_emails_received": sum(e["activity_summary"]["emails_received"] for e in employee_clusters),
    }
    
    # Employee categorization
    developers = []
    managers = []
    communicators = []
    
    for employee in employee_clusters:
        summary = employee["activity_summary"]
        
        # Simple heuristics for categorization
        if summary["git_commits"] > (summary["emails_sent"] + summary["emails_received"]):
            developers.append(employee["employee"])
        elif summary["emails_sent"] > summary["git_commits"] * 2:
            communicators.append(employee["employee"])
            if summary["jira_tickets_assigned"] > 0 and summary["git_commits"] == 0:
                managers.append(employee["employee"])
    
    stats["potential_developers"] = developers
    stats["potential_managers"] = managers
    stats["potential_communicators"] = communicators
    
    # Top contributors
    top_by_commits = sorted(employee_clusters, 
                           key=lambda e: e["activity_summary"]["git_commits"], 
                           reverse=True)[:3]
    
    top_by_jira = sorted(employee_clusters, 
                        key=lambda e: e["activity_summary"]["jira_tickets_resolved"], 
                        reverse=True)[:3]
    
    stats["top_code_contributors"] = [e["employee"] for e in top_by_commits]
    stats["top_issue_resolvers"] = [e["employee"] for e in top_by_jira]
    
    return stats

def identify_collaboration_networks(employee_clusters, jira_data, git_data, email_data):
    """
    Identifies collaboration networks between employees based on shared tasks, 
    emails, and related activities.
    """
    collaboration_graph = defaultdict(lambda: defaultdict(int))
    
    # Step 1: Find collaboration through JIRA tickets
    jira_assignees = {ticket["id"]: ticket.get("assignee") for ticket in jira_data if ticket.get("assignee")}
    
    # Step 2: Find collaboration through email communications
    for email in email_data:
        sender = email.get("sender")
        recipients = email.get("recipients", [])
        
        for recipient in recipients:
            if sender and recipient and sender != recipient:
                collaboration_graph[sender][recipient] += 1
                collaboration_graph[recipient][sender] += 1
    
    # Step 3: Find collaboration through commits to the same JIRA tickets
    ticket_contributors = defaultdict(set)
    for commit in git_data:
        if commit.get("ticket") and commit.get("author"):
            ticket_contributors[commit["ticket"]].add(commit["author"])
    
    for ticket, contributors in ticket_contributors.items():
        # If multiple people contributed to the same ticket
        if len(contributors) > 1:
            contributors_list = list(contributors)
            for i in range(len(contributors_list)):
                for j in range(i+1, len(contributors_list)):
                    collaboration_graph[contributors_list[i]][contributors_list[j]] += 1
                    collaboration_graph[contributors_list[j]][contributors_list[i]] += 1
    
    # Step 4: Format the collaboration data
    collaboration_networks = []
    for employee in collaboration_graph:
        collaborators = [
            {"colleague": colleague, "strength": strength} 
            for colleague, strength in collaboration_graph[employee].items()
        ]
        
        # Sort by collaboration strength
        collaborators.sort(key=lambda x: x["strength"], reverse=True)
        
        collaboration_networks.append({
            "employee": employee,
            "total_collaborations": sum(collaboration_graph[employee].values()),
            "collaborators": collaborators
        })
    
    # Sort by total collaborations
    collaboration_networks.sort(key=lambda x: x["total_collaborations"], reverse=True)
    
    return collaboration_networks

def extract_skills_from_activity(employee_clusters):
    """
    Extract potential skills and expertise areas based on commit messages,
    JIRA tickets, and email content.
    """
    employee_skills = {}
    
    for employee_data in employee_clusters:
        employee = employee_data["employee"]
        skills = defaultdict(int)
        
        # Extract skills from commit messages
        for commit in employee_data["git_commits"]:
            message = commit.get("message", "").lower()
            # Extract technology mentions
            for tech in ["python", "javascript", "react", "node", "api", "database", 
                        "sql", "aws", "docker", "kubernetes", "frontend", "backend"]:
                if tech in message:
                    skills[tech] += 1
            
            # Extract action types
            for action in ["fix", "implement", "refactor", "optimize", "test", "deploy"]:
                if action in message:
                    skills[action] += 1
        
        # Extract from JIRA tickets
        for ticket in employee_data["jira_assigned"] + employee_data["jira_resolved"]:
            summary = ticket.get("summary", "").lower()
            
            # Similar pattern matching for technologies and actions
            for tech in ["python", "javascript", "react", "node", "api", "database", 
                        "sql", "aws", "docker", "kubernetes", "frontend", "backend"]:
                if tech in summary:
                    skills[tech] += 1
        
        # Convert to list of skills with counts
        employee_skills[employee] = [
            {"skill": skill, "mentions": count}
            for skill, count in skills.items()
        ]
        
        # Sort by mention count
        employee_skills[employee].sort(key=lambda x: x["mentions"], reverse=True)
    
    return employee_skills

# Usage example
def process_all_employee_data(jira_data, git_data, email_data):
    """
    Process all employee data and return comprehensive analysis.
    """
    # Create employee clusters
    employee_clusters = cluster_employee_data(jira_data, git_data, email_data)
    
    # Analyze the clusters
    analysis = analyze_employee_clusters(employee_clusters)
    
    # Identify collaboration networks
    collaboration = identify_collaboration_networks(employee_clusters, jira_data, git_data, email_data)
    
    # Extract skills
    skills = extract_skills_from_activity(employee_clusters)
    
    return {
        "employee_clusters": employee_clusters,
        "analysis": analysis,
        "collaboration_networks": collaboration,
        "employee_skills": skills
    }