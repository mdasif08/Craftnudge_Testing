#!/usr/bin/env python3
"""
CraftNudge Service Orchestrator

This script provides comprehensive service management for the CraftNudge microservices:
- Service startup and shutdown
- Health monitoring
- Log aggregation
- Development and production modes
- Service discovery and configuration
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import subprocess
import psutil

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm
from rich.live import Live
from rich.layout import Layout

from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.monitoring.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


class ServiceManager:
    """Manages microservice lifecycle and orchestration."""
    
    def __init__(self):
        self.console = Console()
        self.services = {
            'commit-tracker': {
                'name': 'Commit Tracker Service',
                'port': settings.service.commit_tracker_port,
                'health_url': f'http://localhost:{settings.service.commit_tracker_port}/health',
                'command': ['python', '-m', 'services.commit_tracker.main'],
                'process': None,
                'status': 'stopped'
            },
            'ai-analysis': {
                'name': 'AI Analysis Service',
                'port': settings.service.ai_analysis_port,
                'health_url': f'http://localhost:{settings.service.ai_analysis_port}/health',
                'command': ['python', '-m', 'services.ai_analysis.main'],
                'process': None,
                'status': 'stopped'
            },
            'database-service': {
                'name': 'Database Service',
                'port': settings.service.database_port,
                'health_url': f'http://localhost:{settings.service.database_port}/health',
                'command': ['python', '-m', 'services.database.main'],
                'process': None,
                'status': 'stopped'
            },
            'frontend': {
                'name': 'Frontend Service',
                'port': settings.service.frontend_port,
                'health_url': f'http://localhost:{settings.service.frontend_port}/health',
                'command': ['python', '-m', 'services.frontend.main'],
                'process': None,
                'status': 'stopped'
            },
            'github-webhook': {
                'name': 'GitHub Webhook Service',
                'port': settings.service.github_webhook_port,
                'health_url': f'http://localhost:{settings.service.github_webhook_port}/health',
                'command': ['python', '-m', 'services.github_webhook.main'],
                'process': None,
                'status': 'stopped'
            },
            'commit-quality-coaching': {
                'name': 'Commit Quality Coaching Service',
                'port': settings.service.commit_quality_coaching_port,
                'health_url': f'http://localhost:{settings.service.commit_quality_coaching_port}/health',
                'command': ['python', '-m', 'services.commit_quality_coaching.main'],
                'process': None,
                'status': 'stopped'
            }
        }
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def check_infrastructure(self) -> bool:
        """Check if required infrastructure services are running."""
        infrastructure_checks = [
            ('Redis', 'redis://localhost:6379'),
            ('PostgreSQL', f'postgresql://postgres:password@localhost:5432/craftnudge'),
            ('Ollama', 'http://localhost:11434/api/tags')
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Checking infrastructure...", total=len(infrastructure_checks))
            
            for name, url in infrastructure_checks:
                progress.update(task, description=f"Checking {name}...")
                
                try:
                    if name == 'Redis':
                        import redis
                        r = redis.from_url(url)
                        r.ping()
                    elif name == 'PostgreSQL':
                        import psycopg2
                        conn = psycopg2.connect(url)
                        conn.close()
                    elif name == 'Ollama':
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            response = await client.get(url)
                            if response.status_code != 200:
                                raise Exception(f"Ollama returned status {response.status_code}")
                    
                    progress.advance(task)
                    
                except Exception as e:
                    self.console.print(f"[red]❌ {name} is not available: {e}[/red]")
                    return False
            
            progress.update(task, description="Infrastructure check completed")
        
        self.console.print("[green]✅ All infrastructure services are running[/green]")
        return True
    
    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            self.console.print(f"[red]❌ Unknown service: {service_name}[/red]")
            return False
        
        service = self.services[service_name]
        
        if service['process'] and service['process'].poll() is None:
            self.console.print(f"[yellow]⚠️  {service['name']} is already running[/yellow]")
            return True
        
        try:
            # Start the service
            process = subprocess.Popen(
                service['command'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            service['process'] = process
            service['status'] = 'starting'
            
            # Wait for service to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                self.console.print(f"[red]❌ Failed to start {service['name']}[/red]")
                self.console.print(f"STDOUT: {stdout}")
                self.console.print(f"STDERR: {stderr}")
                return False
            
            # Wait for health check
            for attempt in range(10):
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(service['health_url'])
                        if response.status_code == 200:
                            service['status'] = 'running'
                            self.console.print(f"[green]✅ {service['name']} started successfully[/green]")
                            return True
                except Exception:
                    pass
                
                await asyncio.sleep(1)
            
            service['status'] = 'unhealthy'
            self.console.print(f"[red]❌ {service['name']} failed health check[/red]")
            return False
            
        except Exception as e:
            self.console.print(f"[red]❌ Error starting {service['name']}: {e}[/red]")
            service['status'] = 'error'
            return False
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.services:
            self.console.print(f"[red]❌ Unknown service: {service_name}[/red]")
            return False
        
        service = self.services[service_name]
        
        if not service['process'] or service['process'].poll() is not None:
            self.console.print(f"[yellow]⚠️  {service['name']} is not running[/yellow]")
            return True
        
        try:
            # Send SIGTERM
            service['process'].terminate()
            
            # Wait for graceful shutdown
            try:
                service['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                service['process'].kill()
                service['process'].wait()
            
            service['status'] = 'stopped'
            self.console.print(f"[green]✅ {service['name']} stopped successfully[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Error stopping {service['name']}: {e}[/red]")
            return False
    
    async def start_all_services(self) -> bool:
        """Start all services in the correct order."""
        # Service startup order (dependencies first)
        startup_order = [
            'database-service',
            'commit-tracker',
            'ai-analysis',
            'github-webhook',
            'commit-quality-coaching',
            'frontend'
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Starting services...", total=len(startup_order))
            
            for service_name in startup_order:
                progress.update(task, description=f"Starting {self.services[service_name]['name']}...")
                
                success = await self.start_service(service_name)
                if not success:
                    self.console.print(f"[red]❌ Failed to start {service_name}, stopping all services[/red]")
                    await self.stop_all_services()
                    return False
                
                progress.advance(task)
            
            progress.update(task, description="All services started")
        
        self.console.print("[green]✅ All services started successfully[/green]")
        return True
    
    async def stop_all_services(self) -> bool:
        """Stop all services."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Stopping services...", total=len(self.services))
            
            for service_name in self.services:
                progress.update(task, description=f"Stopping {self.services[service_name]['name']}...")
                await self.stop_service(service_name)
                progress.advance(task)
            
            progress.update(task, description="All services stopped")
        
        self.console.print("[green]✅ All services stopped successfully[/green]")
        return True
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        if service_name not in self.services:
            return {'status': 'unknown', 'error': 'Service not found'}
        
        service = self.services[service_name]
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(service['health_url'])
                if response.status_code == 200:
                    return {
                        'status': 'healthy',
                        'data': response.json(),
                        'response_time': response.elapsed.total_seconds()
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'error': f'HTTP {response.status_code}',
                        'data': response.text
                    }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status of all services."""
        health_status = {}
        
        for service_name in self.services:
            health_status[service_name] = await self.check_service_health(service_name)
        
        return health_status
    
    def display_service_status(self):
        """Display current status of all services."""
        table = Table(title="Service Status", show_header=True, header_style="bold magenta")
        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Port", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("PID", style="blue")
        
        for service_name, service in self.services.items():
            pid = service['process'].pid if service['process'] and service['process'].poll() is None else "N/A"
            
            status_style = {
                'running': 'green',
                'starting': 'yellow',
                'stopped': 'red',
                'error': 'red',
                'unhealthy': 'red'
            }.get(service['status'], 'white')
            
            table.add_row(
                service['name'],
                str(service['port']),
                f"[{status_style}]{service['status']}[/{status_style}]",
                str(pid)
            )
        
        self.console.print(table)
    
    async def monitor_services(self, interval: int = 30):
        """Monitor services and display status updates."""
        self.running = True
        
        def signal_handler(signum, frame):
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while self.running:
            # Clear screen
            self.console.clear()
            
            # Display status
            self.display_service_status()
            
            # Display health status
            health_status = await self.get_all_health_status()
            
            health_table = Table(title="Health Status", show_header=True, header_style="bold magenta")
            health_table.add_column("Service", style="cyan")
            health_table.add_column("Status", style="yellow")
            health_table.add_column("Response Time", style="green")
            health_table.add_column("Details", style="white")
            
            for service_name, health in health_status.items():
                status_style = {
                    'healthy': 'green',
                    'unhealthy': 'red',
                    'error': 'red'
                }.get(health['status'], 'yellow')
                
                response_time = f"{health.get('response_time', 0):.3f}s" if health.get('response_time') else "N/A"
                details = health.get('error', 'OK')
                
                health_table.add_row(
                    self.services[service_name]['name'],
                    f"[{status_style}]{health['status']}[/{status_style}]",
                    response_time,
                    details
                )
            
            self.console.print(health_table)
            
            # Display system info
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            system_info = f"""
System Information:
CPU Usage: {cpu_percent}%
Memory Usage: {memory.percent}% ({memory.used // 1024 // 1024}MB / {memory.total // 1024 // 1024}MB)
            """
            
            self.console.print(Panel(system_info, title="System Info", border_style="blue"))
            
            # Wait for next update
            await asyncio.sleep(interval)


# Service manager instance
service_manager = ServiceManager()


@click.group()
def cli():
    """CraftNudge Service Orchestrator"""
    pass


@cli.command()
@click.option('--service', help='Start specific service only')
@click.option('--check-infrastructure', is_flag=True, help='Check infrastructure before starting')
@click.option('--monitor', is_flag=True, help='Start monitoring after starting services')
def start(service: Optional[str], check_infrastructure: bool, monitor: bool):
    """Start services."""
    async def main():
        # Check infrastructure if requested
        if check_infrastructure:
            if not await service_manager.check_infrastructure():
                console.print("[red]❌ Infrastructure check failed. Please start Redis, PostgreSQL, and Ollama first.[/red]")
                return
        
        if service:
            # Start specific service
            success = await service_manager.start_service(service)
            if not success:
                sys.exit(1)
        else:
            # Start all services
            success = await service_manager.start_all_services()
            if not success:
                sys.exit(1)
        
        if monitor:
            await service_manager.monitor_services()
    
    asyncio.run(main())


@cli.command()
@click.option('--service', help='Stop specific service only')
def stop(service: Optional[str]):
    """Stop services."""
    async def main():
        if service:
            success = await service_manager.stop_service(service)
            if not success:
                sys.exit(1)
        else:
            success = await service_manager.stop_all_services()
            if not success:
                sys.exit(1)
    
    asyncio.run(main())


@cli.command()
def status():
    """Show service status."""
    service_manager.display_service_status()


@cli.command()
@click.option('--interval', default=30, help='Monitoring interval in seconds')
def monitor(interval: int):
    """Monitor services."""
    async def main():
        await service_manager.monitor_services(interval)
    
    asyncio.run(main())


@cli.command()
def restart():
    """Restart all services."""
    async def main():
        console.print("[yellow]Stopping all services...[/yellow]")
        await service_manager.stop_all_services()
        
        console.print("[yellow]Starting all services...[/yellow]")
        success = await service_manager.start_all_services()
        if not success:
            sys.exit(1)
    
    asyncio.run(main())


@cli.command()
def health():
    """Check health of all services."""
    async def main():
        health_status = await service_manager.get_all_health_status()
        
        table = Table(title="Service Health", show_header=True, header_style="bold magenta")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Response Time", style="green")
        table.add_column("Details", style="white")
        
        for service_name, health in health_status.items():
            status_style = {
                'healthy': 'green',
                'unhealthy': 'red',
                'error': 'red'
            }.get(health['status'], 'yellow')
            
            response_time = f"{health.get('response_time', 0):.3f}s" if health.get('response_time') else "N/A"
            details = health.get('error', 'OK')
            
            table.add_row(
                service_manager.services[service_name]['name'],
                f"[{status_style}]{health['status']}[/{status_style}]",
                response_time,
                details
            )
        
        console.print(table)
    
    asyncio.run(main())


@cli.command()
def logs():
    """Show service logs."""
    console.print("[yellow]Service logs feature coming soon![/yellow]")
    console.print("For now, check individual service logs in their respective terminals.")


if __name__ == "__main__":
    cli()
