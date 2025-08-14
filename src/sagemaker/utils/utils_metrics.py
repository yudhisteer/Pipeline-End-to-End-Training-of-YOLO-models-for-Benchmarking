import boto3
import tarfile
import os
import tempfile
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.table import Table
from rich import box
from typing import Dict, Any
import time
import json
from rich.panel import Panel


from utils.utils_config import load_config, get_validation_config, load_inference_config


console = Console()


def get_training_job_details(job_name: str) -> dict:
    """
    Get details for a specific training job.
    
    Args:
        job_name: Name of the training job
        
    Returns:
        Dict containing job details or None if not found
    """
    try:
        sm = boto3.client('sagemaker')
        return sm.describe_training_job(TrainingJobName=job_name)
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] Training job '[yellow]{job_name}[/yellow]' not found: {e}")
        return None


def list_all_training_jobs_with_metrics() -> dict:
    """
    Fetch and display recent training jobs with their metrics in a single table.
    Returns the job details dictionary for the latest completed job.
    """
    console = Console()
    sm = boto3.client('sagemaker')
    
    # Fetch recent training jobs
    console.print("\n[bold blue]📋 Recent Training Jobs with Metrics[/bold blue]", style="bold")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching training jobs and metrics...", total=None)
        all_jobs = sm.list_training_jobs(
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=10
        )
        progress.update(task, completed=True)

    # Create table for training jobs with metrics
    jobs_table = Table(title="Recent Training Jobs with S3 Metrics", box=box.ROUNDED)
    jobs_table.add_column("#", style="cyan", no_wrap=True, width=3)
    jobs_table.add_column("Job Name", style="magenta", min_width=30)
    jobs_table.add_column("Status", justify="center", width=10)
    jobs_table.add_column("Creation Time", style="green", width=16)
    jobs_table.add_column("Best mAP50", style="gold1", justify="right", width=10)
    jobs_table.add_column("Best Recall", style="green", justify="right", width=10)
    jobs_table.add_column("Best Precision", style="blue", justify="right", width=12)
    jobs_table.add_column("Last mAP50", style="yellow", justify="right", width=10)
    jobs_table.add_column("Last Recall", style="dim green", justify="right", width=10)
    jobs_table.add_column("Last Precision", style="dim blue", justify="right", width=12)
    jobs_table.add_column("Duration (s)", style="cyan", justify="right", width=10)
    
    for i, job in enumerate(all_jobs['TrainingJobSummaries']):
        job_name = job['TrainingJobName']
        status_style = "green" if job['TrainingJobStatus'] == 'Completed' else "yellow" if job['TrainingJobStatus'] == 'InProgress' else "red"
        
        # Get metrics for this job if it's completed
        best_map50 = "N/A"
        best_recall = "N/A"
        best_precision = "N/A"
        last_map50 = "N/A"
        last_recall = "N/A"
        last_precision = "N/A"
        duration = "N/A"
        
        if job['TrainingJobStatus'] == 'Completed':
            try:
                job_details = sm.describe_training_job(TrainingJobName=job_name)
                
                # Get duration
                if 'TrainingTimeInSeconds' in job_details:
                    duration = str(job_details['TrainingTimeInSeconds'])
                
                # Extract metrics from S3 artifacts
                try:
                    s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
                    metrics = extract_metrics_from_s3(s3_uri, console)
                    
                    if metrics:
                        # Best epoch metrics
                        if 'best_epoch_metrics' in metrics:
                            best_metrics = metrics['best_epoch_metrics']
                            if 'mAP50' in best_metrics:
                                best_map50 = f"{best_metrics['mAP50']:.4f}"
                            if 'recall' in best_metrics:
                                best_recall = f"{best_metrics['recall']:.4f}"
                            if 'precision' in best_metrics:
                                best_precision = f"{best_metrics['precision']:.4f}"
                        
                        # Last epoch metrics (final epoch)
                        if 'mAP50' in metrics:
                            last_map50 = f"{metrics['mAP50']:.4f}"
                        if 'recall' in metrics:
                            last_recall = f"{metrics['recall']:.4f}"
                        if 'precision' in metrics:
                            last_precision = f"{metrics['precision']:.4f}"
                            
                except Exception as e:
                    # If S3 extraction fails, metrics remain N/A
                    print(f"Debug: S3 extraction failed for {job_name}: {e}")
                    pass
                    
            except Exception as e:
                # If we can't get details, just show N/A for metrics
                pass
        
        jobs_table.add_row(
            str(i+1),
            job_name,
            f"[{status_style}]{job['TrainingJobStatus']}[/{status_style}]",
            job['CreationTime'].strftime("%m-%d %H:%M"),
            best_map50,
            best_recall,
            best_precision,
            last_map50,
            last_recall,
            last_precision,
            duration
        )
    
    console.print(jobs_table)
    
    # Return details of the latest completed job for potential further use
    completed_jobs = [job for job in all_jobs['TrainingJobSummaries'] if job['TrainingJobStatus'] == 'Completed']
    if completed_jobs:
        latest_job_name = completed_jobs[0]['TrainingJobName']
        try:
            return sm.describe_training_job(TrainingJobName=latest_job_name)
        except Exception:
            return None
    
    return None


def list_specific_job_metrics(job_details: dict) -> None:
    """
    Extract and display metrics from S3 artifacts for a given training job.
    
    Args:
        job_details: Dictionary containing training job details from SageMaker
    """
    console = Console()
    
    if not job_details:
        console.print(Panel(
            "[red]❌ No job details provided[/red]",
            title="[red]Error[/red]",
            border_style="red"
        ))
        return
    
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


    
def extract_metrics_from_s3(s3_uri: str, console: Console) -> dict:
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
                    
                    # console.print(archive_table)
                    
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
                            
                            # Find best epoch by mAP50
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




def generate_model_metrics(results, model_dir: str, config: Dict[str, Any] = None):
    """
    Generate validation metrics for SageMaker Model Registry.
    
    Args:
        results: YOLO training results
        model_dir: Directory to save metrics files
        config: Configuration dictionary (optional, will load default if None)
    """
    print("Generating model metrics for SageMaker Model Registry...")
    
    # Load config if not provided
    if config is None:
        try:
            config = load_config()
        except:
            config = {}
    
    # Get validation thresholds from config
    validation_config = get_validation_config(config)
    
    try:
        # Extract metrics from YOLO results
        # YOLO results typically contain validation metrics in the last epoch
        if hasattr(results, 'results_dict'):
            metrics_dict = results.results_dict
        else:
            # Fallback: extract from results object
            metrics_dict = {}
            if hasattr(results, 'metrics'):
                metrics_dict = results.metrics
    
        # YOLO-specific validation metrics for object detection
        validation_metrics = {
            "model_name": "YOLO",
            "validation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": {
                # Primary YOLO metrics
                "mAP_0.5": float(getattr(results, 'maps', [0.0])[0]) if hasattr(results, 'maps') else 0.0,
                "mAP_0.5_0.95": float(getattr(results, 'map', 0.0)) if hasattr(results, 'map') else 0.0,
                "precision": float(getattr(results, 'mp', 0.0)) if hasattr(results, 'mp') else 0.0,
                "recall": float(getattr(results, 'mr', 0.0)) if hasattr(results, 'mr') else 0.0,
                
                # Training metrics
                "final_epoch": int(getattr(results, 'epoch', 0)) if hasattr(results, 'epoch') else 0,
                "best_fitness": float(getattr(results, 'fitness', 0.0)) if hasattr(results, 'fitness') else 0.0,
                
                # Loss metrics
                "box_loss": float(getattr(results, 'box_loss', 0.0)) if hasattr(results, 'box_loss') else 0.0,
                "cls_loss": float(getattr(results, 'cls_loss', 0.0)) if hasattr(results, 'cls_loss') else 0.0,
                "dfl_loss": float(getattr(results, 'dfl_loss', 0.0)) if hasattr(results, 'dfl_loss') else 0.0,
            },
            "model_info": {
                "parameters": getattr(results, 'model_params', 0),
                "model_size_mb": 0.0,  # Will be calculated below
                "inference_speed_ms": 0.0,  # Placeholder for inference speed
            }
        }
        
        # Calculate model size if model files exist
        model_files = ['best.pt', 'last.pt']
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                validation_metrics["model_info"]["model_size_mb"] = round(size_mb, 2)
                break
        
        # Save evaluation metrics
        evaluation_path = os.path.join(model_dir, "evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {evaluation_path}")
        
        # Model constraints (thresholds for model approval) - from config
        model_constraints = {
            "model_quality_constraints": {
                "min_mAP_0.5": validation_config.get('min_mAP_0_5', 0.3),
                "min_precision": validation_config.get('min_precision', 0.5),
                "min_recall": validation_config.get('min_recall', 0.4),
                "max_model_size_mb": validation_config.get('max_model_size_mb', 500),
                "max_inference_time_ms": validation_config.get('max_inference_time_ms', 100)
            },
            "current_performance": {
                "mAP_0.5": validation_metrics["metrics"]["mAP_0.5"],
                "precision": validation_metrics["metrics"]["precision"],
                "recall": validation_metrics["metrics"]["recall"],
                "model_size_mb": validation_metrics["model_info"]["model_size_mb"]
            }
        }
        
        # Save model constraints
        constraints_path = os.path.join(model_dir, "constraints.json")
        with open(constraints_path, 'w') as f:
            json.dump(model_constraints, f, indent=2)
        print(f"Saved model constraints to: {constraints_path}")
        
        # Print summary
        print("Model Validation Summary:")
        print(f"  mAP@0.5: {validation_metrics['metrics']['mAP_0.5']:.3f}")
        print(f"  mAP@0.5:0.95: {validation_metrics['metrics']['mAP_0.5_0.95']:.3f}")
        print(f"  Precision: {validation_metrics['metrics']['precision']:.3f}")
        print(f"  Recall: {validation_metrics['metrics']['recall']:.3f}")
        print(f"  Model Size: {validation_metrics['model_info']['model_size_mb']:.1f} MB")
        
        # Check if model meets quality constraints
        constraints = model_constraints["model_quality_constraints"]
        current = model_constraints["current_performance"]
        
        meets_constraints = (
            current["mAP_0.5"] >= constraints["min_mAP_0.5"] and
            current["precision"] >= constraints["min_precision"] and
            current["recall"] >= constraints["min_recall"] and
            current["model_size_mb"] <= constraints["max_model_size_mb"]
        )
        
        print(f"\nModel Quality Assessment:")
        print(f"  Meets quality constraints: {'✅ YES' if meets_constraints else '❌ NO'}")
        if not meets_constraints:
            print("  Issues found:")
            if current["mAP_0.5"] < constraints["min_mAP_0.5"]:
                print(f"    - mAP@0.5 too low: {current['mAP_0.5']:.3f} < {constraints['min_mAP_0.5']}")
            if current["precision"] < constraints["min_precision"]:
                print(f"    - Precision too low: {current['precision']:.3f} < {constraints['min_precision']}")
            if current["recall"] < constraints["min_recall"]:
                print(f"    - Recall too low: {current['recall']:.3f} < {constraints['min_recall']}")
            if current["model_size_mb"] > constraints["max_model_size_mb"]:
                print(f"    - Model too large: {current['model_size_mb']:.1f} MB > {constraints['max_model_size_mb']} MB")
        
    except Exception as e:
        print(f"Warning: Could not generate complete metrics: {e}")
        # Create minimal metrics file
        minimal_metrics = {
            "model_name": "YOLO",
            "validation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "status": "training_completed",
            "error": str(e)
        }
        
        evaluation_path = os.path.join(model_dir, "evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(minimal_metrics, f, indent=2)


def send_metrics_to_cloudwatch(recall: float, map_50: float, region: str = None) -> bool:
    """
    Send custom metrics to CloudWatch.
    
    Args:
        recall: Model recall value
        map_50: Model mAP@0.5 value  
        region: AWS region (defaults to environment or us-east-1)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Determine region
        aws_region = region or os.environ.get('AWS_REGION', 'us-east-1')
        
        # Create CloudWatch client
        cloudwatch = boto3.client('cloudwatch', region_name=aws_region)
        
        # Get job name for dimensioning
        job_name = os.environ.get('SM_JOB_NAME', 'UnknownJob')
        
        # Prepare metrics
        metrics = [
            {
                'MetricName': 'Recall',
                'Dimensions': [{'Name': 'TrainingJobName', 'Value': job_name}],
                'Value': recall,
                'Unit': 'None'
            },
            {
                'MetricName': 'mAP_0.5',
                'Dimensions': [{'Name': 'TrainingJobName', 'Value': job_name}],
                'Value': map_50,
                'Unit': 'None'
            }
        ]
        
        # Send metrics to CloudWatch
        config = load_inference_config()
        namespace = config['cloudwatch']['namespace']
        cloudwatch.put_metric_data(
            Namespace=namespace,
            MetricData=metrics
        )
        
        print(f"Sent metrics to CloudWatch: recall={recall:.4f}, mAP@0.5={map_50:.4f}")
        return True
        
    except Exception as e:
        print(f"Failed to send metrics to CloudWatch: {e}")
        return False


def save_metrics_for_pipeline(model_dir: str, output_dir: str) -> Dict[str, float]:
    """
    Extract and save metrics from YOLO training results for pipeline consumption.
    
    Args:
        model_dir: Directory containing training results
        output_dir: Directory to save pipeline metrics
        
    Returns:
        Dict containing extracted metrics
    """
    
    # Look for results.csv in training output
    results_csv = os.path.join(model_dir, "results.csv")
    
    if not os.path.exists(results_csv):
        print(f"results.csv not found at {results_csv}")
        raise FileNotFoundError(f"results.csv not found")
    
    # Read results and get final epoch metrics
    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1]  # Last epoch
    
    # Extract key metrics
    recall = float(last_row.get("metrics/recall(B)", 0))
    map_50 = float(last_row.get("metrics/mAP50(B)", 0))
    
    metrics = {
        "recall": recall,
        "map_50": map_50
    }
    
    # Save metrics for pipeline
    os.makedirs(output_dir, exist_ok=True)
    config = load_inference_config()
    metrics_filename = config['registry']['evaluation_metrics_file']
    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Pipeline metrics - recall: {recall:.4f}, mAP@0.5: {map_50:.4f}")
    print(f"Saved metrics to {metrics_path}")
    
    return metrics
