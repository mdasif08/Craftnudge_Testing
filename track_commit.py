#!/usr/bin/env python3
"""
CraftNudge Commit Tracker CLI - User Story 2.1.1 Implementation

This CLI tool provides on-demand Git commit tracking with:
- Real-time commit analysis and logging
- Local JSONL storage for offline tracking
- Comprehensive error handling and user feedback
- Integration with AI analysis services
- Rich console output and progress tracking

Usage:
    python track_commit.py [OPTIONS]

Examples:
    python track_commit.py                           # Track latest commit in current repo
    python track_commit.py --repo-path /path/to/repo # Track latest commit in specific repo
    python track_commit.py --show-latest             # Show latest commit details
    python track_commit.py --recent --days 7         # Track recent commits
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm
from rich.syntax import Syntax

from config.settings import settings
from shared.models import Commit, CommitQuality

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.monitoring.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


class CommitTrackerCLI:
    """CLI interface for commit tracking."""
    
    def __init__(self):
        self.console = Console()
        self.service_url = f"http://localhost:{settings.service.commit_tracker_port}"
    
    async def track_commit_async(
        self, 
        repo_path: str, 
        commit_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track commit via API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.service_url}/track-commit",
                json={
                    "repository_path": repo_path,
                    "commit_hash": commit_hash
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def track_recent_commits_async(
        self, 
        repo_path: str, 
        days: int = 7, 
        limit: int = 100
    ) -> Dict[str, Any]:
        """Track recent commits via API."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.service_url}/track-recent-commits",
                json={
                    "repository_path": repo_path,
                    "days": days,
                    "limit": limit
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def get_commit_statistics_async(self, repo_path: str) -> Dict[str, Any]:
        """Get commit statistics via API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.service_url}/statistics/{repo_path}"
            )
            response.raise_for_status()
            return response.json()
    
    def display_commit_details(self, commit_data: Dict[str, Any]):
        """Display commit details in a rich table."""
        table = Table(title="Commit Details", show_header=True, header_style="bold magenta")
        
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        # Basic commit information
        table.add_row("Hash", commit_data.get("hash", "N/A")[:12] + "...")
        table.add_row("Author", commit_data.get("author", "N/A"))
        table.add_row("Repository", commit_data.get("repository", "N/A"))
        table.add_row("Branch", commit_data.get("branch", "N/A"))
        table.add_row("Timestamp", commit_data.get("timestamp", "N/A"))
        
        # Commit statistics
        additions = commit_data.get("additions", 0)
        deletions = commit_data.get("deletions", 0)
        total_changes = commit_data.get("total_changes", 0)
        
        table.add_row("Additions", f"[green]+{additions}[/green]")
        table.add_row("Deletions", f"[red]-{deletions}[/red]")
        table.add_row("Total Changes", str(total_changes))
        
        # Quality information
        quality_score = commit_data.get("quality_score")
        if quality_score is not None:
            quality_level = commit_data.get("quality_level", "unknown")
            table.add_row("Quality Score", f"{quality_score}/10 ({quality_level})")
        else:
            table.add_row("Quality Score", "Not analyzed yet")
        
        # Changed files
        changed_files = commit_data.get("changed_files", [])
        if changed_files:
            files_text = "\n".join(changed_files[:5])  # Show first 5 files
            if len(changed_files) > 5:
                files_text += f"\n... and {len(changed_files) - 5} more"
            table.add_row("Changed Files", files_text)
        else:
            table.add_row("Changed Files", "No files changed")
        
        # Commit message
        message = commit_data.get("message", "N/A")
        if len(message) > 100:
            message = message[:100] + "..."
        table.add_row("Message", message)
        
        self.console.print(table)
    
    def display_commit_message(self, commit_data: Dict[str, Any]):
        """Display commit message in a syntax-highlighted panel."""
        message = commit_data.get("message", "No message")
        
        # Try to detect language for syntax highlighting
        language = "text"
        if any(keyword in message.lower() for keyword in ["fix", "bug", "issue"]):
            language = "diff"
        elif any(keyword in message.lower() for keyword in ["feat", "feature", "add"]):
            language = "bash"
        
        syntax = Syntax(message, language, theme="monokai", word_wrap=True)
        panel = Panel(syntax, title="Commit Message", border_style="blue")
        self.console.print(panel)
    
    def display_statistics(self, stats_data: Dict[str, Any]):
        """Display repository statistics."""
        stats = stats_data.get("statistics", {})
        
        table = Table(title="Repository Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Total Commits", str(stats.get("total_commits", 0)))
        table.add_row("Total Additions", f"[green]+{stats.get('total_additions', 0)}[/green]")
        table.add_row("Total Deletions", f"[red]-{stats.get('total_deletions', 0)}[/red]")
        table.add_row("Average Commit Size", str(stats.get("average_commit_size", 0)))
        table.add_row("Commit Frequency", f"{stats.get('commit_frequency', 0)} commits/day")
        
        # Top authors
        top_authors = stats.get("top_authors", [])
        if top_authors:
            authors_text = "\n".join([f"{author}: {count}" for author, count in top_authors[:3]])
            table.add_row("Top Authors", authors_text)
        
        # Most changed files
        most_changed_files = stats.get("most_changed_files", [])
        if most_changed_files:
            files_text = "\n".join([f"{file}: {count}" for file, count in most_changed_files[:3]])
            table.add_row("Most Changed Files", files_text)
        
        self.console.print(table)
    
    def display_success_message(self, commit_data: Dict[str, Any]):
        """Display success message with commit information."""
        hash_short = commit_data.get("hash", "")[:8]
        author = commit_data.get("author", "Unknown")
        message = commit_data.get("message", "No message")
        
        if len(message) > 50:
            message = message[:50] + "..."
        
        success_text = Text()
        success_text.append("✅ ", style="bold green")
        success_text.append("Commit tracked successfully!\n\n", style="bold white")
        success_text.append(f"Hash: ", style="cyan")
        success_text.append(f"{hash_short}\n", style="bold white")
        success_text.append(f"Author: ", style="cyan")
        success_text.append(f"{author}\n", style="white")
        success_text.append(f"Message: ", style="cyan")
        success_text.append(f"{message}", style="white")
        
        panel = Panel(success_text, title="Success", border_style="green")
        self.console.print(panel)
    
    def display_error_message(self, error: str, suggestion: str = ""):
        """Display error message with helpful suggestions."""
        error_text = Text()
        error_text.append("❌ ", style="bold red")
        error_text.append("Error occurred\n\n", style="bold white")
        error_text.append(f"Error: ", style="red")
        error_text.append(f"{error}\n", style="white")
        
        if suggestion:
            error_text.append(f"Suggestion: ", style="yellow")
            error_text.append(f"{suggestion}", style="white")
        
        panel = Panel(error_text, title="Error", border_style="red")
        self.console.print(panel)
    
    def display_help_text(self):
        """Display help text with usage examples."""
        help_text = """
[bold cyan]CraftNudge Commit Tracker[/bold cyan]

Track Git commits and analyze your coding patterns with AI-powered insights.

[bold yellow]Usage Examples:[/bold yellow]
  python track_commit.py                           # Track latest commit
  python track_commit.py --repo-path /path/to/repo # Track in specific repository
  python track_commit.py --show-latest             # Show latest commit details
  python track_commit.py --recent --days 7         # Track recent commits

[bold yellow]Features:[/bold yellow]
  ✅ Real-time commit tracking
  ✅ AI-powered quality analysis
  ✅ Local JSONL storage
  ✅ Comprehensive statistics
  ✅ Rich console output

[bold yellow]Data Storage:[/bold yellow]
  Commits are stored locally in: [green]data/behaviors/commits.jsonl[/green]
  Each entry includes unique ID and UTC timestamp
        """
        
        panel = Panel(help_text, title="Help", border_style="blue")
        self.console.print(panel)


# CLI instance
cli = CommitTrackerCLI()


@click.command()
@click.option(
    '--repo-path', 
    default='.', 
    help='Path to Git repository (default: current directory)',
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    '--commit-hash', 
    help='Specific commit hash to track'
)
@click.option(
    '--show-latest', 
    is_flag=True, 
    help='Show latest commit details without tracking'
)
@click.option(
    '--recent', 
    is_flag=True, 
    help='Track recent commits'
)
@click.option(
    '--days', 
    default=7, 
    help='Number of days to look back for recent commits',
    type=click.IntRange(1, 365)
)
@click.option(
    '--limit', 
    default=100, 
    help='Maximum number of commits to track',
    type=click.IntRange(1, 1000)
)
@click.option(
    '--statistics', 
    is_flag=True, 
    help='Show repository statistics'
)
@click.option(
    '--verbose', 
    '-v', 
    is_flag=True, 
    help='Enable verbose output'
)
@click.option(
    '--help', 
    is_flag=True, 
    help='Show detailed help'
)
def track_commit(
    repo_path: str,
    commit_hash: Optional[str],
    show_latest: bool,
    recent: bool,
    days: int,
    limit: int,
    statistics: bool,
    verbose: bool,
    help: bool
):
    """Track Git commits and analyze coding patterns."""
    
    if help:
        cli.display_help_text()
        return
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate repository path
    repo_path = Path(repo_path).resolve()
    if not (repo_path / ".git").exists():
        cli.display_error_message(
            f"Not a Git repository: {repo_path}",
            "Make sure you're in a Git repository or specify the correct path with --repo-path"
        )
        sys.exit(1)
    
    async def main():
        try:
            # Check if service is available
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
                transient=True
            ) as progress:
                task = progress.add_task("Checking service availability...", total=None)
                
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{cli.service_url}/health")
                        if response.status_code != 200:
                            raise Exception("Service not healthy")
                except Exception as e:
                    cli.display_error_message(
                        "Commit tracker service is not available",
                        "Make sure the service is running: python -m services.commit_tracker.main"
                    )
                    sys.exit(1)
                
                progress.update(task, description="Service is available")
            
            if statistics:
                # Show repository statistics
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=cli.console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Fetching repository statistics...", total=None)
                    stats_data = await cli.get_commit_statistics_async(str(repo_path))
                    progress.update(task, description="Statistics retrieved")
                
                cli.display_statistics(stats_data)
                return
            
            if recent:
                # Track recent commits
                if not Confirm.ask(f"Track recent commits from the last {days} days?"):
                    return
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=cli.console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Tracking recent commits...", total=None)
                    result = await cli.track_recent_commits_async(str(repo_path), days, limit)
                    progress.update(task, description="Recent commits tracked")
                
                commits = result.get("commits", [])
                cli.console.print(f"\n[green]✅ Successfully tracked {len(commits)} recent commits[/green]")
                
                # Show summary
                if commits:
                    table = Table(title="Recent Commits Summary")
                    table.add_column("Hash", style="cyan")
                    table.add_column("Author", style="green")
                    table.add_column("Message", style="white")
                    table.add_column("Date", style="yellow")
                    
                    for commit in commits[:10]:  # Show first 10
                        hash_short = commit.get("hash", "")[:8]
                        author = commit.get("author", "Unknown")
                        message = commit.get("message", "No message")
                        if len(message) > 50:
                            message = message[:50] + "..."
                        date = commit.get("timestamp", "Unknown")
                        
                        table.add_row(hash_short, author, message, str(date))
                    
                    cli.console.print(table)
                
                return
            
            if show_latest:
                # Show latest commit details without tracking
                cli.console.print("[yellow]Showing latest commit details (not tracking)...[/yellow]")
                
                # This would require implementing a get-latest-commit endpoint
                # For now, show a placeholder
                cli.console.print("Latest commit details feature coming soon!")
                return
            
            # Track single commit
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
                transient=True
            ) as progress:
                task = progress.add_task("Tracking commit...", total=None)
                result = await cli.track_commit_async(str(repo_path), commit_hash)
                progress.update(task, description="Commit tracked")
            
            # Display results
            cli.display_success_message(result)
            cli.display_commit_details(result)
            cli.display_commit_message(result)
            
            # Show next steps
            cli.console.print("\n[bold cyan]Next Steps:[/bold cyan]")
            cli.console.print("• View commit analysis at: [blue]http://localhost:8000[/blue]")
            cli.console.print("• Track more commits with: [blue]python track_commit.py --recent[/blue]")
            cli.console.print("• View statistics with: [blue]python track_commit.py --statistics[/blue]")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                error_data = e.response.json()
                cli.display_error_message(
                    error_data.get("detail", "Bad request"),
                    "Check your repository path and commit hash"
                )
            elif e.response.status_code == 404:
                cli.display_error_message(
                    "Commit not found",
                    "Make sure the commit hash is correct"
                )
            else:
                cli.display_error_message(
                    f"HTTP error {e.response.status_code}",
                    "Service may be experiencing issues"
                )
            sys.exit(1)
        except httpx.TimeoutException:
            cli.display_error_message(
                "Request timed out",
                "The service may be overloaded or unavailable"
            )
            sys.exit(1)
        except httpx.ConnectError:
            cli.display_error_message(
                "Cannot connect to commit tracker service",
                "Make sure the service is running: python -m services.commit_tracker.main"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            cli.display_error_message(
                str(e),
                "Check the logs for more details"
            )
            sys.exit(1)
    
    # Run the async function
    asyncio.run(main())


if __name__ == "__main__":
    track_commit()
