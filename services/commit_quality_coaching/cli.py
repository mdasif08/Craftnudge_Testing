#!/usr/bin/env python3
"""
Commit Quality Coaching CLI Tool
Part of the Commit Quality Coaching Microservice

This CLI tool provides an interface for AI-powered commit quality coaching.
It communicates with the Commit Quality Coaching Service via HTTP API calls.
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.columns import Columns

from config.settings import Settings

# Initialize Rich console for beautiful output
console = Console()

class CommitQualityCoachingCLI:
    """CLI interface for the Commit Quality Coaching Service."""
    
    def __init__(self):
        self.settings = Settings()
        self.base_url = f"http://localhost:{self.settings.services.commit_quality_coaching.port}"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def coach_commit(self, commit_hash: str, repo_path: Optional[str] = None) -> Dict[str, Any]:
        """Get coaching feedback for a specific commit."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing commit for coaching...", total=None)
                
                # Prepare request data
                data = {"commit_hash": commit_hash}
                if repo_path:
                    data["repo_path"] = repo_path
                
                # Make API call
                response = await self.client.post(
                    f"{self.base_url}/coach-commit",
                    json=data
                )
                
                progress.update(task, completed=True)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_data = response.json()
                    raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                    
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Quality Coaching Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to get coaching feedback: {str(e)}")
    
    async def get_user_progress(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user progress and insights."""
        try:
            params = {}
            if user_id:
                params["user_id"] = user_id
            
            response = await self.client.get(
                f"{self.base_url}/user-progress",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Quality Coaching Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to get user progress: {str(e)}")
    
    async def get_coaching_session(self, session_id: str) -> Dict[str, Any]:
        """Get details of a specific coaching session."""
        try:
            response = await self.client.get(
                f"{self.base_url}/coaching-sessions/{session_id}"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Quality Coaching Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to get coaching session: {str(e)}")
    
    async def create_coaching_session(self, user_id: str, goals: Optional[str] = None) -> Dict[str, Any]:
        """Create a new coaching session."""
        try:
            data = {"user_id": user_id}
            if goals:
                data["goals"] = goals
            
            response = await self.client.post(
                f"{self.base_url}/coaching-sessions",
                json=data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json()
                raise Exception(f"API Error: {error_data.get('detail', 'Unknown error')}")
                
        except httpx.ConnectError:
            raise Exception("Could not connect to Commit Quality Coaching Service. Is it running?")
        except Exception as e:
            raise Exception(f"Failed to create coaching session: {str(e)}")

def display_coaching_feedback(feedback_data: Dict[str, Any]):
    """Display coaching feedback in a beautiful format."""
    console.print("\n")
    
    # Create main panel
    title = Text("🎯 Commit Quality Coaching", style="bold green")
    content = f"""
    🆔 Session ID: {feedback_data.get('session_id', 'N/A')}
    🔗 Commit Hash: {feedback_data.get('commit_hash', 'N/A')}
    📊 Quality Score: {feedback_data.get('quality_score', 'N/A')}
    🎯 Overall Rating: {feedback_data.get('overall_rating', 'N/A')}
    """
    
    console.print(Panel(content, title=title, border_style="green"))
    
    # Display feedback sections
    feedback = feedback_data.get('feedback', {})
    
    # Message Quality
    message_feedback = feedback.get('message_quality', {})
    if message_feedback:
        console.print(Panel(
            f"📝 {message_feedback.get('rating', 'N/A')}/10 - {message_feedback.get('comment', 'No comment')}",
            title="💬 Commit Message Quality",
            border_style="blue"
        ))
    
    # File Changes
    file_feedback = feedback.get('file_changes', {})
    if file_feedback:
        console.print(Panel(
            f"📁 {file_feedback.get('rating', 'N/A')}/10 - {file_feedback.get('comment', 'No comment')}",
            title="📁 File Changes Quality",
            border_style="yellow"
        ))
    
    # Commit Size
    size_feedback = feedback.get('commit_size', {})
    if size_feedback:
        console.print(Panel(
            f"📏 {size_feedback.get('rating', 'N/A')}/10 - {size_feedback.get('comment', 'No comment')}",
            title="📏 Commit Size Quality",
            border_style="magenta"
        ))
    
    # Suggestions
    suggestions = feedback_data.get('suggestions', [])
    if suggestions:
        console.print(Panel(
            "\n".join([f"💡 {suggestion}" for suggestion in suggestions]),
            title="💡 Improvement Suggestions",
            border_style="cyan"
        ))
    
    # Next Steps
    next_steps = feedback_data.get('next_steps', [])
    if next_steps:
        console.print(Panel(
            "\n".join([f"🎯 {step}" for step in next_steps]),
            title="🎯 Recommended Next Steps",
            border_style="red"
        ))

def display_user_progress(progress_data: Dict[str, Any]):
    """Display user progress in a comprehensive format."""
    console.print("\n")
    
    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    # Header with overall stats
    header_content = f"""
    📊 Overall Progress: {progress_data.get('overall_progress', 'N/A')}%
    🎯 Average Quality Score: {progress_data.get('average_quality_score', 'N/A')}
    📈 Total Commits Analyzed: {progress_data.get('total_commits', 'N/A')}
    """
    layout["header"].update(Panel(header_content, title="📈 User Progress Overview", border_style="green"))
    
    # Body with detailed metrics
    metrics = progress_data.get('metrics', {})
    if metrics:
        metric_content = ""
        for category, value in metrics.items():
            metric_content += f"• {category}: {value}\n"
        
        layout["body"].update(Panel(metric_content, title="📊 Detailed Metrics", border_style="blue"))
    
    # Footer with insights
    insights = progress_data.get('insights', [])
    if insights:
        insight_content = "\n".join([f"🧠 {insight}" for insight in insights])
        layout["footer"].update(Panel(insight_content, title="🧠 AI-Generated Insights", border_style="yellow"))
    
    console.print(layout)

def display_coaching_session(session_data: Dict[str, Any]):
    """Display coaching session details."""
    console.print("\n")
    
    title = Text("🎓 Coaching Session", style="bold blue")
    content = f"""
    🆔 Session ID: {session_data.get('session_id', 'N/A')}
    👤 User ID: {session_data.get('user_id', 'N/A')}
    📅 Created: {session_data.get('created_at', 'N/A')}
    🎯 Goals: {session_data.get('goals', 'No specific goals set')}
    📊 Progress: {session_data.get('progress', 'N/A')}%
    """
    
    console.print(Panel(content, title=title, border_style="blue"))
    
    # Display session activities
    activities = session_data.get('activities', [])
    if activities:
        table = Table(title="📋 Session Activities", show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan")
        table.add_column("Activity", style="yellow")
        table.add_column("Score", style="green")
        
        for activity in activities:
            table.add_row(
                activity.get('date', 'N/A'),
                activity.get('description', 'N/A'),
                str(activity.get('score', 'N/A'))
            )
        
        console.print(table)

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Commit Quality Coaching CLI - Get AI-powered coaching for your commits."""
    pass

@cli.command()
@click.argument('commit_hash', type=str)
@click.option('--repo-path', '-p', help='Path to Git repository (default: current directory)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def coach(commit_hash: str, repo_path: Optional[str], verbose: bool):
    """Get coaching feedback for a specific commit."""
    async def run():
        try:
            async with CommitQualityCoachingCLI() as cli_tool:
                feedback_data = await cli_tool.coach_commit(commit_hash, repo_path)
                display_coaching_feedback(feedback_data)
                
                if verbose:
                    console.print(f"\n[dim]Raw data: {json.dumps(feedback_data, indent=2)}[/dim]")
                    
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
@click.option('--user-id', '-u', help='User ID (default: current user)')
def progress(user_id: Optional[str]):
    """Display user progress and insights."""
    async def run():
        try:
            async with CommitQualityCoachingCLI() as cli_tool:
                progress_data = await cli_tool.get_user_progress(user_id)
                display_user_progress(progress_data)
                
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
@click.argument('session_id', type=str)
def session(session_id: str):
    """Get details of a specific coaching session."""
    async def run():
        try:
            async with CommitQualityCoachingCLI() as cli_tool:
                session_data = await cli_tool.get_coaching_session(session_id)
                display_coaching_session(session_data)
                
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
@click.argument('user_id', type=str)
@click.option('--goals', '-g', help='Coaching goals')
def create_session(user_id: str, goals: Optional[str]):
    """Create a new coaching session."""
    async def run():
        try:
            async with CommitQualityCoachingCLI() as cli_tool:
                session_data = await cli_tool.create_coaching_session(user_id, goals)
                display_coaching_session(session_data)
                
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

@cli.command()
def status():
    """Check the status of the Commit Quality Coaching Service."""
    async def run():
        try:
            async with CommitQualityCoachingCLI() as cli_tool:
                response = await cli_tool.client.get(f"{cli_tool.base_url}/health")
                
                if response.status_code == 200:
                    console.print("[green]✅ Commit Quality Coaching Service is running[/green]")
                    health_data = response.json()
                    console.print(f"[dim]Status: {health_data.get('status', 'unknown')}[/dim]")
                else:
                    console.print("[red]❌ Commit Quality Coaching Service is not responding[/red]")
                    
        except Exception as e:
            console.print(f"[red]❌ Error: {str(e)}[/red]")
            sys.exit(1)
    
    asyncio.run(run())

if __name__ == "__main__":
    cli()
