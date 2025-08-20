#!/usr/bin/env python3
"""
Commit Quality Coaching CLI

This CLI tool provides AI-powered coaching and feedback for commit quality,
helping developers improve their coding practices and commit habits.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich import box

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


class CommitQualityCoachingCLI:
    """CLI for commit quality coaching service."""
    
    def __init__(self):
        self.service_url = "http://localhost:8005"
        self.console = Console()
    
    async def get_coaching_feedback_async(self, commit_id: str, user_id: str, include_context: bool = True) -> Dict[str, Any]:
        """Get coaching feedback for a specific commit."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.service_url}/coach/commit",
                    json={
                        "commit_id": commit_id,
                        "user_id": user_id,
                        "include_historical_context": include_context
                    }
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting coaching feedback: {e}")
            raise
    
    async def get_user_progress_async(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user progress over time."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.service_url}/coach/progress/{user_id}",
                    params={"days": days}
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting user progress: {e}")
            raise
    
    async def get_user_insights_async(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get AI-generated insights for a user."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.service_url}/coach/insights/{user_id}",
                    params={"days": days}
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error getting user insights: {e}")
            raise
    
    async def start_coaching_session_async(self, user_id: str) -> Dict[str, Any]:
        """Start a new coaching session."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.service_url}/coach/session/start",
                    params={"user_id": user_id}
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error starting coaching session: {e}")
            raise
    
    def display_coaching_feedback(self, feedback_data: Dict[str, Any]):
        """Display coaching feedback in a rich format."""
        feedback = feedback_data.get("feedback", {})
        
        # Create main feedback panel
        quality_score = feedback.get("quality_score", 0)
        quality_level = feedback.get("quality_level", "UNKNOWN")
        
        # Color code based on quality
        if quality_score >= 8.0:
            score_color = "green"
            level_color = "green"
        elif quality_score >= 6.0:
            score_color = "yellow"
            level_color = "yellow"
        else:
            score_color = "red"
            level_color = "red"
        
        # Header with quality score
        header = Text()
        header.append("🎯 ", style="bold blue")
        header.append(f"Commit Quality Score: ", style="bold")
        header.append(f"{quality_score:.1f}/10", style=f"bold {score_color}")
        header.append(f" ({quality_level})", style=f"bold {level_color}")
        
        self.console.print(Panel(header, title="Coaching Feedback", border_style="blue"))
        
        # Strengths
        strengths = feedback.get("strengths", [])
        if strengths:
            self.console.print("\n[bold green]✅ Strengths:[/bold green]")
            for strength in strengths:
                self.console.print(f"  • {strength}")
        
        # Areas for improvement
        areas = feedback.get("areas_for_improvement", [])
        if areas:
            self.console.print("\n[bold yellow]⚠️  Areas for Improvement:[/bold yellow]")
            for area in areas:
                self.console.print(f"  • {area}")
        
        # Specific recommendations
        recommendations = feedback.get("specific_recommendations", [])
        if recommendations:
            self.console.print("\n[bold blue]💡 Specific Recommendations:[/bold blue]")
            for rec in recommendations:
                self.console.print(f"  • {rec}")
        
        # Coaching tips
        tips = feedback.get("coaching_tips", [])
        if tips:
            self.console.print("\n[bold cyan]🎓 Coaching Tips:[/bold cyan]")
            for tip in tips:
                self.console.print(f"  • {tip}")
        
        # Next steps
        next_steps = feedback.get("next_steps", [])
        if next_steps:
            self.console.print("\n[bold magenta]🚀 Next Steps:[/bold magenta]")
            for step in next_steps:
                self.console.print(f"  • {step}")
    
    def display_user_progress(self, progress_data: Dict[str, Any]):
        """Display user progress in a rich format."""
        self.console.print(Panel("📊 User Progress Report", title="Progress Analysis", border_style="green"))
        
        # Basic stats
        total_commits = progress_data.get("total_commits", 0)
        avg_score = progress_data.get("average_quality_score", 0.0)
        consistency = progress_data.get("consistency_score", 0.0)
        
        stats_table = Table(title="Progress Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan", no_wrap=True)
        stats_table.add_column("Value", style="green")
        stats_table.add_column("Status", style="yellow")
        
        stats_table.add_row("Total Commits", str(total_commits), "📈" if total_commits > 0 else "📉")
        stats_table.add_row("Average Quality Score", f"{avg_score:.1f}/10", 
                           "🟢" if avg_score >= 7.0 else "🟡" if avg_score >= 5.0 else "🔴")
        stats_table.add_row("Consistency Score", f"{consistency:.1f}/10",
                           "🟢" if consistency >= 7.0 else "🟡" if consistency >= 5.0 else "🔴")
        
        self.console.print(stats_table)
        
        # Quality distribution
        distribution = progress_data.get("quality_distribution", {})
        if distribution:
            self.console.print("\n[bold]Quality Distribution:[/bold]")
            dist_table = Table(box=box.SIMPLE)
            dist_table.add_column("Quality Level", style="cyan")
            dist_table.add_column("Count", style="green")
            dist_table.add_column("Percentage", style="yellow")
            
            total = sum(distribution.values())
            for level, count in distribution.items():
                percentage = (count / total * 100) if total > 0 else 0
                dist_table.add_row(level, str(count), f"{percentage:.1f}%")
            
            self.console.print(dist_table)
        
        # Improvement areas
        improvements = progress_data.get("improvement_areas", [])
        if improvements:
            self.console.print("\n[bold green]📈 Areas Showing Improvement:[/bold green]")
            for area in improvements:
                self.console.print(f"  • {area}")
        
        # Regression areas
        regressions = progress_data.get("regression_areas", [])
        if regressions:
            self.console.print("\n[bold red]📉 Areas Showing Regression:[/bold red]")
            for area in regressions:
                self.console.print(f"  • {area}")
    
    def display_user_insights(self, insights_data: Dict[str, Any]):
        """Display user insights in a rich format."""
        self.console.print(Panel("🧠 AI-Generated Insights", title="Behavior Analysis", border_style="purple"))
        
        # Key insights
        key_insights = insights_data.get("key_insights", [])
        if key_insights:
            self.console.print("\n[bold blue]🔍 Key Insights:[/bold blue]")
            for insight in key_insights:
                self.console.print(f"  • {insight}")
        
        # Recommendations
        recommendations = insights_data.get("recommendations", [])
        if recommendations:
            self.console.print("\n[bold green]💡 Recommendations:[/bold green]")
            for rec in recommendations:
                self.console.print(f"  • {rec}")
        
        # Trends
        trends = insights_data.get("trends", {})
        if trends:
            self.console.print("\n[bold yellow]📈 Trends:[/bold yellow]")
            for metric, trend in trends.items():
                trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
                self.console.print(f"  • {metric.replace('_', ' ').title()}: {trend} {trend_icon}")
    
    def display_error_message(self, title: str, message: str):
        """Display error message in a rich format."""
        error_text = Text()
        error_text.append("❌ ", style="bold red")
        error_text.append(title, style="bold red")
        error_text.append("\n\n", style="red")
        error_text.append(message, style="red")
        
        self.console.print(Panel(error_text, title="Error", border_style="red"))
    
    def display_help_text(self):
        """Display help text."""
        help_text = """
[bold blue]Commit Quality Coaching CLI[/bold blue]

This tool provides AI-powered coaching and feedback for your commit quality,
helping you improve your coding practices and commit habits.

[bold green]Available Commands:[/bold green]

• [cyan]coach-commit[/cyan] - Get coaching feedback for a specific commit
• [cyan]progress[/cyan] - View your progress over time
• [cyan]insights[/cyan] - Get AI-generated insights about your commit patterns
• [cyan]session[/cyan] - Start a coaching session

[bold yellow]Examples:[/bold yellow]

  python coach_commit.py coach-commit --commit-id abc123 --user-id john
  python coach_commit.py progress --user-id john --days 30
  python coach_commit.py insights --user-id john --days 30

[bold magenta]Tips:[/bold magenta]

• Use the --include-context flag to get historical context
• Set up regular coaching sessions to track your improvement
• Review insights weekly to identify patterns
• Focus on one improvement area at a time
        """
        
        self.console.print(Panel(help_text, title="Help", border_style="blue"))


# Initialize CLI
cli = CommitQualityCoachingCLI()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--help', is_flag=True, help='Show detailed help')
def main(verbose: bool, help: bool):
    """Commit Quality Coaching CLI - AI-powered coaching for better commits."""
    
    if help:
        cli.display_help_text()
        return
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option('--commit-id', required=True, help='ID of the commit to coach')
@click.option('--user-id', required=True, help='ID of the user requesting coaching')
@click.option('--include-context', is_flag=True, default=True, help='Include historical context')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def coach_commit(commit_id: str, user_id: str, include_context: bool, verbose: bool):
    """Get coaching feedback for a specific commit."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_coaching():
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
                        "Commit quality coaching service is not available",
                        "Make sure the service is running: python -m services.commit_quality_coaching.main"
                    )
                    sys.exit(1)
                
                progress.update(task, description="Service is available")
            
            # Get coaching feedback
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
                transient=True
            ) as progress:
                task = progress.add_task("Analyzing commit and generating feedback...", total=None)
                feedback_data = await cli.get_coaching_feedback_async(commit_id, user_id, include_context)
                progress.update(task, description="Feedback generated")
            
            # Display results
            cli.display_coaching_feedback(feedback_data)
            
            # Show next steps
            cli.console.print("\n[bold cyan]Next Steps:[/bold cyan]")
            cli.console.print("• Review the feedback and implement recommendations")
            cli.console.print("• Track your progress over time with: [blue]python coach_commit.py progress[/blue]")
            cli.console.print("• Get insights about your patterns with: [blue]python coach_commit.py insights[/blue]")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                cli.display_error_message(
                    "Commit not found",
                    "Make sure the commit ID is correct and the commit has been tracked"
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
                "Cannot connect to commit quality coaching service",
                "Make sure the service is running: python -m services.commit_quality_coaching.main"
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
    asyncio.run(run_coaching())


@main.command()
@click.option('--user-id', required=True, help='ID of the user to get progress for')
@click.option('--days', default=30, help='Number of days to look back', type=click.IntRange(1, 365))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def progress(user_id: str, days: int, verbose: bool):
    """View user progress over time."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_progress():
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
                        "Commit quality coaching service is not available",
                        "Make sure the service is running: python -m services.commit_quality_coaching.main"
                    )
                    sys.exit(1)
                
                progress.update(task, description="Service is available")
            
            # Get user progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
                transient=True
            ) as progress:
                task = progress.add_task("Analyzing user progress...", total=None)
                progress_data = await cli.get_user_progress_async(user_id, days)
                progress.update(task, description="Progress analysis completed")
            
            # Display results
            cli.display_user_progress(progress_data)
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                cli.display_error_message(
                    "User progress not found",
                    "Make sure the user ID is correct and the user has tracked commits"
                )
            else:
                cli.display_error_message(
                    f"HTTP error {e.response.status_code}",
                    "Service may be experiencing issues"
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
    asyncio.run(run_progress())


@main.command()
@click.option('--user-id', required=True, help='ID of the user to get insights for')
@click.option('--days', default=30, help='Number of days to look back', type=click.IntRange(1, 365))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def insights(user_id: str, days: int, verbose: bool):
    """Get AI-generated insights about user commit patterns."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_insights():
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
                        "Commit quality coaching service is not available",
                        "Make sure the service is running: python -m services.commit_quality_coaching.main"
                    )
                    sys.exit(1)
                
                progress.update(task, description="Service is available")
            
            # Get user insights
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
                transient=True
            ) as progress:
                task = progress.add_task("Generating AI insights...", total=None)
                insights_data = await cli.get_user_insights_async(user_id, days)
                progress.update(task, description="Insights generated")
            
            # Display results
            cli.display_user_insights(insights_data)
            
        except httpx.HTTPStatusError as e:
            cli.display_error_message(
                f"HTTP error {e.response.status_code}",
                "Service may be experiencing issues"
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
    asyncio.run(run_insights())


@main.command()
@click.option('--user-id', required=True, help='ID of the user to start session for')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def session(user_id: str, verbose: bool):
    """Start a coaching session."""
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async def run_session():
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
                        "Commit quality coaching service is not available",
                        "Make sure the service is running: python -m services.commit_quality_coaching.main"
                    )
                    sys.exit(1)
                
                progress.update(task, description="Service is available")
            
            # Start coaching session
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=cli.console,
                transient=True
            ) as progress:
                task = progress.add_task("Starting coaching session...", total=None)
                session_data = await cli.start_coaching_session_async(user_id)
                progress.update(task, description="Session started")
            
            # Display session info
            session_id = session_data.get("session_id", "unknown")
            start_time = session_data.get("start_time", "unknown")
            
            cli.console.print(Panel(
                f"Session ID: [bold cyan]{session_id}[/bold cyan]\n"
                f"Start Time: [bold green]{start_time}[/bold green]\n\n"
                f"Your coaching session has started! Use this session ID to track your progress.",
                title="Coaching Session Started",
                border_style="green"
            ))
            
        except httpx.HTTPStatusError as e:
            cli.display_error_message(
                f"HTTP error {e.response.status_code}",
                "Service may be experiencing issues"
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
    asyncio.run(run_session())


if __name__ == "__main__":
    main()

