import boto3
import json
import sys
import tempfile
import tarfile
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.columns import Columns
import pandas as pd

def display_training_job_metrics(training_job_name: str):
    """
    Standalone function to display metrics for a specific training job.
    
    Args:
        training_job_name: Name of the SageMaker training job
    """
    console = Console()
    
    # Display banner
    console.print(Panel(
        "[bold blue]SageMaker Training Job Metrics Extractor[/bold blue]\n"
        "[dim]Fetches and displays training metrics from SageMaker jobs[/dim]",
        title="[bold green]🚀 YOLO Training Metrics[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))
    
    console.print(f"[bold cyan]🎯 Displaying metrics for:[/bold cyan] [yellow]{training_job_name}[/yellow]\n")
    
    get_training_job_metrics(training_job_name)

def get_training_job_metrics(specific_job_name: str = None) -> None:
    console = Console()
    sm = boto3.client('sagemaker')
    
    # Only list jobs if no specific job is provided
    if not specific_job_name:
        # First, let's list all recent training jobs
        console.print("\n[bold blue]📋 Recent Training Jobs[/bold blue]", style="bold")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching training jobs...", total=None)
            all_jobs = sm.list_training_jobs(
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=10
            )
            progress.update(task, completed=True)

        # Create table for training jobs
        jobs_table = Table(title="Recent Training Jobs", box=box.ROUNDED)
        jobs_table.add_column("#", style="cyan", no_wrap=True, width=3)
        jobs_table.add_column("Job Name", style="magenta", min_width=30)
        jobs_table.add_column("Status", justify="center", width=12)
        jobs_table.add_column("Creation Time", style="green", width=20)
        
        for i, job in enumerate(all_jobs['TrainingJobSummaries']):
            status_style = "green" if job['TrainingJobStatus'] == 'Completed' else "yellow" if job['TrainingJobStatus'] == 'InProgress' else "red"
            jobs_table.add_row(
                str(i+1),
                job['TrainingJobName'],
                f"[{status_style}]{job['TrainingJobStatus']}[/{status_style}]",
                job['CreationTime'].strftime("%Y-%m-%d %H:%M:%S")
            )
        
        console.print(jobs_table)

    # Determine which job to analyze
    if specific_job_name:
        console.print(f"\n[bold cyan]🎯 Using specified job:[/bold cyan] [yellow]{specific_job_name}[/yellow]")
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching job details...", total=None)
                job_details = sm.describe_training_job(TrainingJobName=specific_job_name)
                job_name = specific_job_name
                progress.update(task, completed=True)
        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] Training job '[yellow]{specific_job_name}[/yellow]' not found: {e}")
            return
    else:
        # Find the latest COMPLETED job (all_jobs was fetched above)
        completed_jobs = [job for job in all_jobs['TrainingJobSummaries'] if job['TrainingJobStatus'] == 'Completed']
        
        if not completed_jobs:
            console.print("\n[bold red]❌ No completed training jobs found![/bold red]")
            return
        
        job_name = completed_jobs[0]['TrainingJobName']  # Most recent completed job
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching job details...", total=None)
            job_details = sm.describe_training_job(TrainingJobName=job_name)
            progress.update(task, completed=True)
        
        # Find the position in the full list
        job_position = next(i+1 for i, job in enumerate(all_jobs['TrainingJobSummaries']) if job['TrainingJobName'] == job_name)
        console.print(f"\n[bold green]✅ Using latest completed job[/bold green] [cyan](#{job_position})[/cyan]: [yellow]{job_name}[/yellow]")

    # Check for final metrics
    final_metrics = job_details.get('FinalMetricDataList', [])
    if final_metrics:
        metrics_table = Table(title="Final Metrics from SageMaker", box=box.ROUNDED)
        metrics_table.add_column("Metric Name", style="cyan")
        metrics_table.add_column("Value", style="green", justify="right")
        
        for metric in final_metrics:
            metrics_table.add_row(metric["MetricName"], str(metric["Value"]))
        console.print(metrics_table)
    else:
        console.print(Panel(
            "[yellow]No final metrics found in SageMaker training job[/yellow]\n"
            "[dim]Note: Metrics may be in the training logs or saved to S3[/dim]",
            title="[red]No Metrics Available[/red]",
            border_style="yellow"
        ))
    
    # Training job summary
    job_info = [
        f"[cyan]Job Name:[/cyan] {job_name}",
        f"[cyan]Status:[/cyan] [green]{job_details['TrainingJobStatus']}[/green]",
        f"[cyan]Training Time:[/cyan] {job_details.get('TrainingStartTime', 'N/A')} to {job_details.get('TrainingEndTime', 'N/A')}",
        f"[cyan]Duration:[/cyan] {job_details.get('TrainingTimeInSeconds', 'N/A')} seconds",
        f"[cyan]Billable Seconds:[/cyan] {job_details.get('BillableTimeInSeconds', 'N/A')}",
        f"[cyan]Model Artifacts:[/cyan] [dim]{job_details['ModelArtifacts']['S3ModelArtifacts']}[/dim]"
    ]
    
    console.print(Panel(
        "\n".join(job_info),
        title="[bold blue]📊 Training Job Summary[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Try to extract metrics from S3 artifacts
    console.print(f'\n[bold yellow]🔍 Attempting to extract metrics from S3 artifacts...[/bold yellow]')
    try:
        s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
        metrics = extract_metrics_from_s3(s3_uri, console)
        
        if metrics:
            console.print(f'\n[bold green]📊 Extracted Training Metrics[/bold green]')
            
            # Create metrics table for final epoch
            final_metrics_table = Table(title="Final Epoch Metrics", box=box.ROUNDED)
            final_metrics_table.add_column("Metric", style="cyan")
            final_metrics_table.add_column("Value", style="green", justify="right")
            
            # Add final epoch metrics (excluding best_epoch_metrics)
            for key, value in metrics.items():
                if key != 'best_epoch_metrics':
                    if isinstance(value, float):
                        final_metrics_table.add_row(key, f"{value:.4f}")
                    else:
                        final_metrics_table.add_row(key, str(value))
            
            # Create best epoch metrics table
            if 'best_epoch_metrics' in metrics:
                best_metrics = metrics['best_epoch_metrics']
                best_metrics_table = Table(title="🏆 Best Epoch Metrics (Highest mAP50)", box=box.ROUNDED)
                best_metrics_table.add_column("Metric", style="cyan")
                best_metrics_table.add_column("Value", style="gold1", justify="right")
                
                for key, value in best_metrics.items():
                    if isinstance(value, float):
                        best_metrics_table.add_row(key, f"{value:.4f}")
                    else:
                        best_metrics_table.add_row(key, str(value))
                
                # Display tables one after another
                console.print(final_metrics_table)
                console.print(best_metrics_table)
            else:
                console.print(final_metrics_table)
        else:
            console.print(Panel(
                f"[yellow]⚠️ Could not extract metrics from S3 artifacts[/yellow]\n"
                f"[dim]📋 Manual extraction required from:[/dim]\n[blue]{s3_uri}[/blue]",
                title="[red]Extraction Failed[/red]",
                border_style="red"
            ))
    except Exception as e:
        console.print(Panel(
            f"[red]Error extracting metrics: {e}[/red]\n"
            f"[dim]Check S3 artifacts manually:[/dim]\n[blue]{job_details['ModelArtifacts']['S3ModelArtifacts']}[/blue]",
            title="[red]❌ Extraction Error[/red]",
            border_style="red"
        ))

def extract_metrics_from_s3(s3_uri, console):
    """Extract evaluation_metrics.json from S3 model artifacts"""
    try:
        s3 = boto3.client('s3')
        
        # Parse S3 URI
        bucket = s3_uri.replace("s3://", "").split("/")[0]
        key = "/".join(s3_uri.replace("s3://", "").split("/")[1:])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Download task
            download_task = progress.add_task(f"📥 Downloading from s3://{bucket}/{key}", total=None)
            
            # Download the tar.gz file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                s3.download_file(bucket, key, tmp_file.name)
                progress.update(download_task, completed=True)
                console.print(f"  [green]📦 Downloaded {os.path.getsize(tmp_file.name):,} bytes[/green]")
                
                # Extract task
                extract_task = progress.add_task("🔍 Extracting and searching files...", total=None)
                
                # Extract and look for results.csv (contains training metrics)
                with tarfile.open(tmp_file.name, 'r:gz') as tar:
                    progress.update(extract_task, completed=True)
                    
                    # Create table for archive contents
                    archive_table = Table(title=f"Archive Contents ({len(tar.getmembers())} files)", box=box.MINIMAL)
                    archive_table.add_column("File", style="cyan")
                    archive_table.add_column("Type", style="yellow", width=10)
                    
                    results_file = None
                    for member in tar.getmembers():
                        file_type = "📁 Dir" if member.isdir() else "📄 File"
                        archive_table.add_row(member.name, file_type)
                        
                        # Look for results.csv which contains the actual training metrics
                        if member.name.endswith('results.csv'):
                            results_file = member
                    
                    console.print(archive_table)
                    
                    if results_file:
                        console.print(f"  [bold green]✅ Found results file:[/bold green] [yellow]{results_file.name}[/yellow]")
                        extracted_file = tar.extractfile(results_file)
                        if extracted_file:
                            # Parse CSV and get last row (final epoch metrics)
                            import io
                            
                            csv_content = extracted_file.read().decode('utf-8')
                            df = pd.read_csv(io.StringIO(csv_content))
                            
                            last_row = df.iloc[-1]  # Last epoch (final metrics)
                            
                            # Extract key metrics from final epoch
                            metrics_data = {
                                'epoch': int(last_row.get('epoch', 0)),
                                'precision': float(last_row.get('metrics/precision(B)', 0)),
                                'recall': float(last_row.get('metrics/recall(B)', 0)),
                                'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                                'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                                'train_box_loss': float(last_row.get('train/box_loss', 0)),
                                'val_box_loss': float(last_row.get('val/box_loss', 0))
                            }
                            
                            # Find best epoch by mAP50 (since fitness column doesn't exist)
                            best_map_idx = df['metrics/mAP50(B)'].idxmax()
                            best_row = df.iloc[best_map_idx]
                            
                            # Create best epoch metrics for comparison
                            best_metrics = {
                                'epoch': int(best_row.get('epoch', 0)),
                                'precision': float(best_row.get('metrics/precision(B)', 0)),
                                'recall': float(best_row.get('metrics/recall(B)', 0)),
                                'mAP50': float(best_row.get('metrics/mAP50(B)', 0)),
                                'mAP50-95': float(best_row.get('metrics/mAP50-95(B)', 0)),
                                'train_box_loss': float(best_row.get('train/box_loss', 0)),
                                'val_box_loss': float(best_row.get('val/box_loss', 0))
                            }
                            
                            # Store both metrics for return
                            metrics_data['best_epoch_metrics'] = best_metrics
                            
                            console.print(f"\n  [bold gold1]🏆 Best Epoch (highest mAP50): {int(best_row.get('epoch', 0))}[/bold gold1]")
                            
                            os.unlink(tmp_file.name)  # Clean up
                            return metrics_data
                    else:
                        console.print("[red]No results.csv found in archive[/red]")
                        os.unlink(tmp_file.name)  # Clean up
                        return None
            
    except Exception as e:
        console.print(f"[red]Error extracting metrics from S3: {e}[/red]")
        return None

if __name__ == "__main__":
    console = Console()
    
    # Display banner
    console.print(Panel(
        "[bold blue]SageMaker Training Job Metrics Extractor[/bold blue]\n"
        "[dim]Fetches and displays training metrics from SageMaker jobs[/dim]",
        title="[bold green]🚀 YOLO Training Metrics[/bold green]",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Check if specific job name provided as command line argument
    if len(sys.argv) > 1:
        specific_job = sys.argv[1]
        console.print(f"[bold yellow]💡 Usage:[/bold yellow] python {sys.argv[0]} [training_job_name]")
        console.print(f"[bold cyan]🔍 Looking for job:[/bold cyan] [yellow]{specific_job}[/yellow]\n")
        display_training_job_metrics(specific_job)
    else:
        console.print(f"[bold yellow]💡 Usage:[/bold yellow] python {sys.argv[0]} [training_job_name]")
        console.print(f"[bold cyan]📋 No job specified, using latest completed job[/bold cyan]\n")
        get_training_job_metrics()


# Example usage:
"""
YOLO SageMaker Training Metrics - Usage Examples

Command Line Usage:
# Display metrics for specific training job
python src/sagemaker/sagemaker_metrics.py pipelines-lwyromzfdl9t-YOLOTrainingStep-931HVQyPKu

# Display metrics for latest completed job
python src/sagemaker/sagemaker_metrics.py

"""