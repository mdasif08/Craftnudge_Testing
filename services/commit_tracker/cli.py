#!/usr/bin/env python3
"""
Commit Tracker CLI Tool
Part of the Commit Tracker Microservice

This CLI tool provides an interface for tracking Git commits and analyzing
commit behavior patterns. It communicates with the Commit Tracker Service
via HTTP API calls.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import Settings

# Initialize Rich console for beautiful output
console = Console()

class CommitTrackerCLI:
    """CLI interface for the Commit Tracker Service."""
    
    def __init__(self):
        self.settings = Settings()
        self.base_url = f"http://localhost:{self.settings.services.commit_tracker.port}"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def track_commit(self, repo_path: Optional[str] = None) -> Dict[str, Any]:
        """Track the latest commit in the repository."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Tracking commit...", total=None)
                
                # Prepare request data
                data = {}
                if repo_path:
                    data["repo_path"] = repo_path
                
                # Make API call
                response = await self.client.post(
                    f"{self.base_url}/track-commit",
                    json=data
                )
                
                progress.update(task, completed=True)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_data = response.json()
                    raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                    
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Tracker Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to track commit: {str(e)}")
    
    async def get_commit_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get commit history from the service."""
        try:
            response = await self.client.get(
                f"{self.base_url}/commits",
                params={"limit": limit}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Tracker Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to get commit history: {str(e)}")
    
    async def get_commit_analysis(self, commit_id: str) -> Dict[str, Any]:
        """Get analysis for a specific commit."""
        try:
            response = await self.client.get(
                f"{self.base_url}/commits/{commit_id}/analysis"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Tracker Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to get commit analysis: {str(e)}")

def display_commit_info(commit_data: Dict[str, Any]):
    """Display commit information in a beautiful format."""
    console.print("\n")
    
    # Create main panel
    title = Text("📝 Commit Tracked Successfully", style="bold green")
    content = f"""
    🆔 Commit ID: {commit_data.get('id', 'N/A')}
    🔗 Hash: {commit_data.get('hash', 'N/A')}
    👤 Author: {commit_data.get('author', 'N/A')}
    📅 Timestamp: {commit_data.get('timestamp', 'N/A')}
    📊 Quality Score: {commit_data.get('quality_score', 'N/A')}
    """
    
    console.print(Panel(content, title=title, border_style="green"))
    
    # Display commit message
    message = commit_data.get('message', 'No message')
    console.print(Panel(message, title="💬 Commit Message", border_style="blue"))
    
    # Display changed files
    changed_files = commit_data.get('changed_files', [])
    if changed_files:
        table = Table(title="📁 Changed Files", show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="yellow")
        
        for file_info in changed_files:
            table.add_row(
                file_info.get('path', 'N/A'),
                file_info.get('status', 'N/A')
            )
        
        console.print(table)
    
    # Display analysis if available
    analysis = commit_data.get('analysis', {})
    if analysis:
        console.print(Panel(
            f"🔍 Analysis: {analysis.get('summary', 'No analysis available')}",
            title="🧠 AI Analysis",
            border_style="yellow"
        ))

def display_commit_history(history_data: Dict[str, Any]):
    """Display commit history in a table format."""
    commits = history_data.get('commits', [])
    
    if not commits:
        console.print(Panel("No commits found in history.", title="📋 Commit History"))
        return
    
    table = Table(title="📋 Recent Commits", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Hash", style="green", width=12)
    table.add_column("Author", style="yellow", width=15)
    table.add_column("Message", style="white", width=40)
    table.add_column("Quality", style="blue", width=10)
    table.add_column("Date", style="red", width=20)
    
    for commit in commits:
        table.add_row(
            str(commit.get('id', 'N/A')),
            commit.get('hash', 'N/A')[:8] + '...',
            commit.get('author', 'N/A')[:12] + '...' if len(commit.get('author', '')) > 12 else commit.get('author', 'N/A'),
            commit.get('message', 'N/A')[:37] + '...' if len(commit.get('message', '')) > 37 else commit.get('message', 'N/A'),
            str(commit.get('quality_score', 'N/A')),
            commit.get('timestamp', 'N/A')[:19] if commit.get('timestamp') else 'N/A'
        )
    
    console.print(table)

def display_analysis(analysis_data: Dict[str, Any]):
    """Display commit analysis in a detailed format."""
    console.print("\n")
    
    title = Text("🧠 Commit Analysis", style="bold blue")
    content = f"""
    📊 Quality Score: {analysis_data.get('quality_score', 'N/A')}
    🎯 Quality Level: {analysis_data.get('quality_level', 'N/A')}
    📝 Summary: {analysis_data.get('summary', 'N/A')}
    💡 Suggestions: {analysis_data.get('suggestions', 'N/A')}
    🔍 Patterns Detected: {analysis_data.get('patterns_detected', 'N/A')}
    🧠 Behavioral Insights: {analysis_data.get('behavioral_insights', 'N/A')}
    ⏱️ Analysis Duration: {analysis_data.get('analysis_duration', 'N/A')}ms
    🤖 Model Used: {analysis_data.get('model_used', 'N/A')}
    """
    
    console.print(Panel(content, title=title, border_style="blue"))

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Commit Tracker CLI - Track and analyze Git commits."""
    pass

@cli.command()
@click.option('--repo-path', '-p', help='Path to Git repository (default: current directory)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def track(repo_path: Optional[str], verbose: bool):
    """Track the latest commit in the repository."""
    async def run():
        try:
            async with CommitTrackerCLI() as cli_tool:
                commit_data = await cli_tool.track_commit(repo_path)
                display_commit_info(commit_data)
                
                if verbose:
                    console.print(f"\n[dim]Raw data: {json.dumps(commit_data, indent=2)}[/dim]")
                    
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
@click.option('--limit', '-l', default=10, help='Number of commits to display (default: 10)')
def history(limit: int):
    """Display recent commit history."""
    async def run():
        try:
            async with CommitTrackerCLI() as cli_tool:
                history_data = await cli_tool.get_commit_history(limit)
                display_commit_history(history_data)
                
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
@click.argument('commit_id', type=str)
def analyze(commit_id: str):
    """Analyze a specific commit by ID."""
    async def run():
        try:
            async with CommitTrackerCLI() as cli_tool:
                analysis_data = await cli_tool.get_commit_analysis(commit_id)
                display_analysis(analysis_data)
                
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
def status():
    """Check the status of the Commit Tracker Service."""
    async def run():
        try:
            async with CommitTrackerCLI() as cli_tool:
                response = await cli_tool.client.get(f"{cli_tool.base_url}/health")
                
                if response.status_code == 200:
                    console.print("[green]✅ Commit Tracker Service is running[/green]")
                    health_data = response.json()
                    console.print(f"[dim]Status: {health_data.get('status', 'unknown')}[/dim]")
                else:
                    console.print("[red]❌ Commit Tracker Service is not responding[/red]")
                    
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

if __name__ == "__main__":
    cli()
