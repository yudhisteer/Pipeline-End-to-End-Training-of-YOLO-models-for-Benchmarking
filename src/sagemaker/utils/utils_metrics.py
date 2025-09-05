import io
import os
import tarfile
import tempfile
import time
import json
from typing import Dict, Any

import boto3
import pandas as pd
from rich import print
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


from utils.utils_config import (
    get_validation_config,
    get_approval_config,
    load_config,
    get_inference_config
)

def get_training_job_model_path(job_name: str) -> str:
    """
    Get the model path for a specific training job.
    """
    job_details = get_training_job_details(job_name)
    
    if job_details is None:
        raise ValueError(f"Training job '{job_name}' not found or could not be retrieved")
    
    if "ModelArtifacts" not in job_details:
        raise ValueError(f"Training job '{job_name}' does not have ModelArtifacts")
    
    if "S3ModelArtifacts" not in job_details["ModelArtifacts"]:
        raise ValueError(f"Training job '{job_name}' does not have S3ModelArtifacts")
    
    return job_details["ModelArtifacts"]["S3ModelArtifacts"]


def get_training_job_details(job_name: str) -> dict:
    """
    Get details for a specific training job.

    Args:
        job_name: Name of the training job

    Returns:
        Dict containing job details or None if not found
    """
    console = Console()
    try:
        sm = boto3.client("sagemaker")
        return sm.describe_training_job(TrainingJobName=job_name)
    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] Training job '[yellow]{job_name}[/yellow]' not found: {e}"
        )
        return None


def list_training_jobs(max_results: int = 10) -> dict:
    """
    Fetch and display recent training jobs with basic information only (no metrics).

    Args:
        max_results: Maximum number of training jobs to fetch and display

    Returns:
        Job details dictionary for the latest completed job.
    """
    console = Console()
    sm = boto3.client("sagemaker")

    console.print("\n[bold blue]Recent Training Jobs[/bold blue]", style="bold")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching training jobs...", total=None)
        all_jobs = sm.list_training_jobs(
            SortBy="CreationTime", SortOrder="Descending", MaxResults=max_results
        )
        progress.update(task, completed=True)

    jobs_table = Table(title="Recent Training Jobs", box=box.ROUNDED)
    jobs_table.add_column("#", style="cyan", no_wrap=True, width=3)
    jobs_table.add_column("Job Name", style="magenta", min_width=30)
    jobs_table.add_column("Status", justify="center", width=12)
    jobs_table.add_column("Creation Time", style="green", width=16)
    jobs_table.add_column("Duration (s)", style="cyan", justify="right", width=12)

    for i, job in enumerate(all_jobs["TrainingJobSummaries"]):
        job_name = job["TrainingJobName"]
        status_style = (
            "green"
            if job["TrainingJobStatus"] == "Completed"
            else "yellow" if job["TrainingJobStatus"] == "InProgress" else "red"
        )
        duration = "N/A"
        if job["TrainingJobStatus"] == "Completed":
            try:
                job_details = sm.describe_training_job(TrainingJobName=job_name)
                if "TrainingTimeInSeconds" in job_details:
                    duration = str(job_details["TrainingTimeInSeconds"])
            except Exception:
                pass

        jobs_table.add_row(
            str(i + 1),
            job_name,
            f"[{status_style}]{job['TrainingJobStatus']}[/{status_style}]",
            job["CreationTime"].strftime("%m-%d %H:%M"),
            duration,
        )

    console.print(jobs_table)

    completed_jobs = [
        job
        for job in all_jobs["TrainingJobSummaries"]
        if job["TrainingJobStatus"] == "Completed"
    ]
    if completed_jobs:
        latest_job_name = completed_jobs[0]["TrainingJobName"]
        try:
            return sm.describe_training_job(TrainingJobName=latest_job_name)
        except Exception:
            return None

    return None


def list_all_training_jobs_with_metrics(max_results: int = 10) -> dict:
    """
    Fetch and display recent training jobs with their metrics in a single table.

    Args:
        max_results: Maximum number of training jobs to fetch and display

    Returns:
        Job details dictionary for the latest completed job.
    """
    console = Console()
    sm = boto3.client("sagemaker")

    console.print(
        "\n[bold blue]Recent Training Jobs with Metrics[/bold blue]", style="bold"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching training jobs and metrics...", total=None)
        all_jobs = sm.list_training_jobs(
            SortBy="CreationTime", SortOrder="Descending", MaxResults=max_results
        )
        progress.update(task, completed=True)

    jobs_table = Table(title="Recent Training Jobs with S3 Metrics", box=box.ROUNDED)
    jobs_table.add_column("#", style="cyan", no_wrap=True, width=3)
    jobs_table.add_column("Job Name", style="magenta", width=55, no_wrap=False, overflow="ellipsis")
    jobs_table.add_column("Status", justify="center", width=10)
    jobs_table.add_column("Creation Time", style="green", width=16)
    jobs_table.add_column("Best mAP50", style="gold1", justify="right", width=10)
    jobs_table.add_column("Best Recall", style="green", justify="right", width=10)
    jobs_table.add_column("Best Precision", style="blue", justify="right", width=12)
    jobs_table.add_column("Last mAP50", style="yellow", justify="right", width=10)
    jobs_table.add_column("Last Recall", style="bright_green", justify="right", width=10)
    jobs_table.add_column("Last Precision", style="bright_cyan", justify="right", width=12)
    jobs_table.add_column("Duration (s)", style="cyan", justify="right", width=10)

    for i, job in enumerate(all_jobs["TrainingJobSummaries"]):
        job_name = job["TrainingJobName"]
        status_style = (
            "green"
            if job["TrainingJobStatus"] == "Completed"
            else "yellow" if job["TrainingJobStatus"] == "InProgress" else "red"
        )

        # get metrics for this job if it's completed
        best_map50 = "N/A"
        best_recall = "N/A"
        best_precision = "N/A"
        last_map50 = "N/A"
        last_recall = "N/A"
        last_precision = "N/A"
        duration = "N/A"

        if job["TrainingJobStatus"] == "Completed":
            try:
                job_details = sm.describe_training_job(TrainingJobName=job_name)

                if "TrainingTimeInSeconds" in job_details:
                    duration = str(job_details["TrainingTimeInSeconds"])

                try:
                    s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
                    metrics = extract_metrics_from_s3(s3_uri, console)

                    if metrics:
                        # best epoch metrics
                        if "best_epoch_metrics" in metrics:
                            best_metrics = metrics["best_epoch_metrics"]
                            if "mAP50" in best_metrics:
                                best_map50 = f"{best_metrics['mAP50']:.4f}"
                            if "recall" in best_metrics:
                                best_recall = f"{best_metrics['recall']:.4f}"
                            if "precision" in best_metrics:
                                best_precision = f"{best_metrics['precision']:.4f}"

                        # last epoch metrics
                        if "mAP50" in metrics:
                            last_map50 = f"{metrics['mAP50']:.4f}"
                        if "recall" in metrics:
                            last_recall = f"{metrics['recall']:.4f}"
                        if "precision" in metrics:
                            last_precision = f"{metrics['precision']:.4f}"

                except Exception as e:
                    print(f"Debug: S3 extraction failed for {job_name}: {e}")
                    pass

            except Exception as e:
                # if we can't get details, just show N/A for metrics
                pass

        jobs_table.add_row(
            str(i + 1),
            job_name,
            f"[{status_style}]{job['TrainingJobStatus']}[/{status_style}]",
            job["CreationTime"].strftime("%m-%d %H:%M"),
            best_map50,
            best_recall,
            best_precision,
            last_map50,
            last_recall,
            last_precision,
            duration,
        )

    console.print(jobs_table)

    completed_jobs = [
        job
        for job in all_jobs["TrainingJobSummaries"]
        if job["TrainingJobStatus"] == "Completed"
    ]
    if completed_jobs:
        latest_job_name = completed_jobs[0]["TrainingJobName"]
        try:
            return sm.describe_training_job(TrainingJobName=latest_job_name)
        except Exception:
            return None

    return None


def list_specific_job_with_metrics(job_details: dict) -> None:
    """
    Extract and display metrics from S3 artifacts for a given training job.

    Args:
        job_details: Dictionary containing training job details from SageMaker
    """
    console = Console()

    if not job_details:
        console.print(
            Panel(
                "[red]No job details provided[/red]",
                title="[red]Error[/red]",
                border_style="red",
            )
        )
        return

    console.print(
        f"\n[bold yellow]Extracting metrics from S3 artifacts...[/bold yellow]"
    )
    try:
        s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
        metrics = extract_metrics_from_s3(s3_uri, console)

        if metrics:
            console.print(f"\n[bold green]Extracted Training Metrics[/bold green]")

            # create metrics table for final epoch
            final_metrics_table = Table(title="Final Epoch Metrics", box=box.ROUNDED)
            final_metrics_table.add_column("Metric", style="cyan")
            final_metrics_table.add_column("Value", style="green", justify="right")

            # add final epoch metrics (excluding best_epoch_metrics)
            for key, value in metrics.items():
                if key != "best_epoch_metrics":
                    if isinstance(value, float):
                        final_metrics_table.add_row(key, f"{value:.4f}")
                    else:
                        final_metrics_table.add_row(key, str(value))

            # create best epoch metrics table
            if "best_epoch_metrics" in metrics:
                best_metrics = metrics["best_epoch_metrics"]
                best_metrics_table = Table(
                    title="Best Epoch Metrics (Highest mAP50)", box=box.ROUNDED
                )
                best_metrics_table.add_column("Metric", style="cyan")
                best_metrics_table.add_column("Value", style="gold1", justify="right")

                for key, value in best_metrics.items():
                    if isinstance(value, float):
                        best_metrics_table.add_row(key, f"{value:.4f}")
                    else:
                        best_metrics_table.add_row(key, str(value))

                console.print(final_metrics_table)
                console.print(best_metrics_table)
            else:
                console.print(final_metrics_table)
        else:
            console.print(
                Panel(
                    f"[yellow]Could not extract metrics from S3 artifacts[/yellow]\n"
                    f"[dim]Manual extraction required from:[/dim]\n[blue]{s3_uri}[/blue]",
                    title="[red]Extraction Failed[/red]",
                    border_style="red",
                )
            )
    except Exception as e:
        console.print(
            Panel(
                f"[red]Error extracting metrics: {e}[/red]\n"
                f"[dim]Check S3 artifacts manually:[/dim]\n[blue]{job_details['ModelArtifacts']['S3ModelArtifacts']}[/blue]",
                title="[red]Extraction Error[/red]",
                border_style="red",
            )
        )


def extract_metrics_from_s3(s3_uri: str, console: Console) -> dict:
    """Extract evaluation_metrics.json from S3 model artifacts"""
    try:
        s3 = boto3.client("s3")

        bucket = s3_uri.replace("s3://", "").split("/")[0]
        key = "/".join(s3_uri.replace("s3://", "").split("/")[1:])

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            download_task = progress.add_task(
                f"Downloading from s3://{bucket}/{key}", total=None
            )

            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", delete=False
            ) as tmp_file:
                s3.download_file(bucket, key, tmp_file.name)
                progress.update(download_task, completed=True)

                extract_task = progress.add_task(
                    "Extracting and searching files...", total=None
                )

                with tarfile.open(tmp_file.name, "r:gz") as tar:
                    progress.update(extract_task, completed=True)

                    # create table for archive contents
                    archive_table = Table(
                        title=f"Archive Contents ({len(tar.getmembers())} files)",
                        box=box.MINIMAL,
                    )
                    archive_table.add_column("File", style="cyan")
                    archive_table.add_column("Type", style="yellow", width=10)

                    results_file = None
                    for member in tar.getmembers():
                        file_type = "Dir" if member.isdir() else "File"
                        archive_table.add_row(member.name, file_type)

                        # look for results.csv which contains the actual training metrics
                        if member.name.endswith("results.csv"):
                            results_file = member

                    # console.print(archive_table)

                    if results_file:
                        console.print(
                            f"  [bold green]Found results file:[/bold green] [yellow]{results_file.name}[/yellow]"
                        )
                        extracted_file = tar.extractfile(results_file)
                        if extracted_file:

                            csv_content = extracted_file.read().decode("utf-8")
                            df = pd.read_csv(io.StringIO(csv_content))

                            last_row = df.iloc[-1]

                            # extract key metrics from final epoch
                            metrics_data = {
                                "epoch": int(last_row.get("epoch", 0)),
                                "precision": float(last_row.get("metrics/precision(B)", 0)),
                                "recall": float(last_row.get("metrics/recall(B)", 0)),
                                "mAP50": float(last_row.get("metrics/mAP50(B)", 0)),
                                "mAP50-95": float(last_row.get("metrics/mAP50-95(B)", 0)),
                                "train_box_loss": float(last_row.get("train/box_loss", 0)),
                                "val_box_loss": float(last_row.get("val/box_loss", 0)),
                            }

                            # find best epoch by mAP50
                            best_map_idx = df["metrics/mAP50(B)"].idxmax()
                            best_row = df.iloc[best_map_idx]

                            # create best epoch metrics for comparison
                            best_metrics = {
                                "epoch": int(best_row.get("epoch", 0)),
                                "precision": float(best_row.get("metrics/precision(B)", 0)),
                                "recall": float(best_row.get("metrics/recall(B)", 0)),
                                "mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
                                "mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0)),
                                "train_box_loss": float(best_row.get("train/box_loss", 0)),
                                "val_box_loss": float(best_row.get("val/box_loss", 0)),
                            }

                            metrics_data["best_epoch_metrics"] = best_metrics

                            console.print(
                                f"\n  [bold gold1]Best Epoch (highest mAP50): \
                                    {int(best_row.get('epoch', 0))}[/bold gold1]"
                            )

                            os.unlink(tmp_file.name)
                            return metrics_data
                    else:
                        console.print("[red]No results.csv found in archive[/red]")
                        os.unlink(tmp_file.name)
                        return None

    except Exception as e:
        console.print(f"[red]Error extracting metrics from S3: {e}[/red]")
        return None


def generate_model_metrics(results: dict, model_dir: str, config: Dict[str, Any] = None, training_job_name: str = None) -> None:
    """
    Generate validation metrics for SageMaker Model Registry.
    
    Args:
        results: YOLO training results
        model_dir: Directory to save metrics files
        config: Configuration dictionary (optional, will load default if None)
        training_job_name: Optional training job name to extract metrics from S3
    """
    print("Generating model metrics for SageMaker Model Registry...")
    
    # Load config if not provided
    if config is None:
        try:
            config = load_config()
        except:
            config = {}
    
    # Get quality gates from approval config
    approval_config = get_approval_config(config)
    quality_gates_config = approval_config.get('quality_gates', {})
    
    try:
        # Try S3 extraction first if training_job_name is provided
        if training_job_name:
            try:
                # Get training job details to extract S3 URI
                sagemaker_client = boto3.client('sagemaker')
                job_details = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
                s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
                
                # Use the same extraction method as the display table
                console = Console()
                metrics_data = extract_metrics_from_s3(s3_uri, console)
                
                if metrics_data:
                    mAP_0_5 = float(metrics_data.get("mAP50", 0.0))
                    mAP_0_5_0_95 = float(metrics_data.get("mAP50-95", 0.0))
                    precision = float(metrics_data.get("precision", 0.0))
                    recall = float(metrics_data.get("recall", 0.0))
                    epoch = int(metrics_data.get("epoch", 0))
                    box_loss = float(metrics_data.get("train_box_loss", 0.0))
                    cls_loss = 0.0  # Not available in S3 data
                    dfl_loss = 0.0  # Not available in S3 data
                else:
                    raise Exception("No metrics data extracted from S3")
                    
            except Exception as s3_error:
                print(f"S3 extraction failed: {s3_error}, falling back to local methods")
                training_job_name = None  # Force fallback
        
        if not training_job_name:
            # Extract metrics from results.csv file (more reliable than results object)
            results_csv = os.path.join(model_dir, "results.csv")
            
            if os.path.exists(results_csv):
                # Read metrics from CSV file (same source as the display table)
                df = pd.read_csv(results_csv)
                last_row = df.iloc[-1]
                
                # Extract key metrics using the same method as save_metrics_for_pipeline
                mAP_0_5 = float(last_row.get("metrics/mAP50(B)", 0.0))
                mAP_0_5_0_95 = float(last_row.get("metrics/mAP50-95(B)", 0.0))
                precision = float(last_row.get("metrics/precision(B)", 0.0))
                recall = float(last_row.get("metrics/recall(B)", 0.0))
                epoch = int(last_row.get("epoch", 0))
                
                # Try to extract loss metrics
                box_loss = float(last_row.get("train/box_loss", 0.0))
                cls_loss = float(last_row.get("train/cls_loss", 0.0))
                dfl_loss = float(last_row.get("train/dfl_loss", 0.0))
            else:
                # Fallback: extract from results object (less reliable)
                print(f"Warning: results.csv not found at {results_csv}, using results object")
                mAP_0_5 = float(getattr(results, 'maps', [0.0])[0]) if hasattr(results, 'maps') else 0.0
                mAP_0_5_0_95 = float(getattr(results, 'map', 0.0)) if hasattr(results, 'map') else 0.0
                precision = float(getattr(results, 'mp', 0.0)) if hasattr(results, 'mp') else 0.0
                recall = float(getattr(results, 'mr', 0.0)) if hasattr(results, 'mr') else 0.0
                epoch = int(getattr(results, 'epoch', 0)) if hasattr(results, 'epoch') else 0
                box_loss = float(getattr(results, 'box_loss', 0.0)) if hasattr(results, 'box_loss') else 0.0
                cls_loss = float(getattr(results, 'cls_loss', 0.0)) if hasattr(results, 'cls_loss') else 0.0
                dfl_loss = float(getattr(results, 'dfl_loss', 0.0)) if hasattr(results, 'dfl_loss') else 0.0
    
        # YOLO-specific model metrics for object detection
        model_metrics = {
            "model_name": "YOLO",
            "validation_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "metrics": {
                # Primary YOLO metrics - using config naming convention
                "mAP_0_5": mAP_0_5,
                "mAP_0_5_0_95": mAP_0_5_0_95,
                "precision": precision,
                "recall": recall,
                
                # Training metrics
                "final_epoch": epoch,
                "best_fitness": float(getattr(results, 'fitness', 0.0)) if hasattr(results, 'fitness') else 0.0,
                
                # Loss metrics
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "dfl_loss": dfl_loss,
            },
            "model_info": {
                "parameters": getattr(results, 'model_params', 0),
                "model_size_mb": 0.0,  # Will be calculated below
                "inference_speed_ms": 0.0,  # Placeholder for inference speed
            }
        }
        
        # Calculate model size if model files exist (ONNX format)
        model_files = ['best.onnx']
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                model_metrics["model_info"]["model_size_mb"] = round(size_mb, 2)
                break
        
        # Save evaluation metrics
        evaluation_path = os.path.join(model_dir, "evaluation.json")
        with open(evaluation_path, 'w') as f:
            json.dump(model_metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {evaluation_path}")
        
        # Model constraints (thresholds for model approval) - from config
        model_constraints = {
            "model_quality_constraints": quality_gates_config,
            "current_performance": {
                "mAP_0_5": model_metrics["metrics"]["mAP_0_5"],
                "precision": model_metrics["metrics"]["precision"],
                "recall": model_metrics["metrics"]["recall"],
                "model_size_mb": model_metrics["model_info"]["model_size_mb"]
            }
        }
        
        # Save model constraints
        constraints_path = os.path.join(model_dir, "constraints.json")
        with open(constraints_path, 'w') as f:
            json.dump(model_constraints, f, indent=2)
        print(f"Saved model constraints to: {constraints_path}")
        
        # Print summary
        print("Model Validation Summary:")
        print(f"  mAP@0.5: {model_metrics['metrics']['mAP_0_5']:.3f}")
        print(f"  mAP@0.5:0.95: {model_metrics['metrics']['mAP_0_5_0_95']:.3f}")
        print(f"  Precision: {model_metrics['metrics']['precision']:.3f}")
        print(f"  Recall: {model_metrics['metrics']['recall']:.3f}")
        print(f"  Model Size: {model_metrics['model_info']['model_size_mb']:.1f} MB")
        
        # Check if model meets quality constraints
        constraints = model_constraints["model_quality_constraints"]
        current = model_constraints["current_performance"]
        
        # Check constraints dynamically based on what's available in config
        meets_constraints = True
        constraint_checks = []
        
        # Check each constraint that exists in config
        if "min_mAP_0_5" in constraints and "mAP_0_5" in current:
            check = current["mAP_0_5"] >= constraints["min_mAP_0_5"]
            constraint_checks.append(check)
            meets_constraints = meets_constraints and check
            
        if "min_precision" in constraints and "precision" in current:
            check = current["precision"] >= constraints["min_precision"]
            constraint_checks.append(check)
            meets_constraints = meets_constraints and check
            
        if "min_recall" in constraints and "recall" in current:
            check = current["recall"] >= constraints["min_recall"]
            constraint_checks.append(check)
            meets_constraints = meets_constraints and check
            
        if "max_model_size_mb" in constraints and "model_size_mb" in current:
            check = current["model_size_mb"] <= constraints["max_model_size_mb"]
            constraint_checks.append(check)
            meets_constraints = meets_constraints and check
        
        print(f"\nModel Quality Assessment:")
        print(f"  Meets quality constraints: {'YES' if meets_constraints else 'NO'}")
        if not meets_constraints:
            print("  Issues found:")
            # Check each constraint dynamically
            if "min_mAP_0_5" in constraints and "mAP_0_5" in current:
                if current["mAP_0_5"] < constraints["min_mAP_0_5"]:
                    print(f"    - mAP@0.5 too low: {current['mAP_0_5']:.3f} < {constraints['min_mAP_0_5']}")
            if "min_precision" in constraints and "precision" in current:
                if current["precision"] < constraints["min_precision"]:
                    print(f"    - Precision too low: {current['precision']:.3f} < {constraints['min_precision']}")
            if "min_recall" in constraints and "recall" in current:
                if current["recall"] < constraints["min_recall"]:
                    print(f"    - Recall too low: {current['recall']:.3f} < {constraints['min_recall']}")
            if "max_model_size_mb" in constraints and "model_size_mb" in current:
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
        full_config = load_config()
        config = get_inference_config(full_config)
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
    
    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1] 
    
    # Extract key metrics
    recall = float(last_row.get("metrics/recall(B)", 0))
    map_50 = float(last_row.get("metrics/mAP50(B)", 0))
    
    metrics = {
        "recall": recall,
        "map_50": map_50
    }
    
    # Save metrics for pipeline
    os.makedirs(output_dir, exist_ok=True)
    full_config = load_config()
    config = get_inference_config(full_config)
    metrics_filename = config['registry']['evaluation_metrics_file']
    metrics_path = os.path.join(output_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Pipeline metrics - recall: {recall:.4f}, mAP@0.5: {map_50:.4f}")
    print(f"Saved metrics to {metrics_path}")
    
    return metrics



def get_training_job_config(training_job_name: str) -> str:
    """
    Retrieve the config.yaml used during a training job from S3.
    Focuses on pipeline training configs: {prefix}/config/{timestamp}-{execution_id}/config.yaml
    
    Args:
        training_job_name: Name of the training job
        
    Returns:
        Config content as string, or error message if not found
    """
    try:
        # Extract execution ID from training job name
        # Training job names typically follow pattern: pipelines-{execution_id}-{step_name}-{suffix}
        if not training_job_name.startswith('pipelines-'):
            return "Training job name doesn't follow expected pipeline pattern (should start with 'pipelines-')"
        
        # Extract execution ID (part between 'pipelines-' and the next '-')
        parts = training_job_name.split('-')
        if len(parts) < 2:
            return "Could not extract execution ID from training job name"
        
        execution_id = parts[1]  # e.g., "fnthdyhhsm1z"
        print(f"Extracted execution ID: {execution_id}")
        
        # Load AWS config
        try:
            from utils.utils_config import load_config, get_aws_config
            config = load_config("config.yaml")
            aws_config = get_aws_config(config)
        except Exception:
            # Fallback to default values
            aws_config = {
                'bucket': 'yolo-pipeline-bucket',  # This will likely fail, but user will see error
                'prefix': 'yolo-pipeline'
            }
        
        bucket = aws_config.get('bucket')
        prefix = aws_config.get('prefix', 'yolo-pipeline')
        
        # Search for config with the specific execution ID
        s3_client = boto3.client('s3')
        
        # Search pattern: {prefix}/config/*-{execution_id}/config.yaml
        config_prefix = f"{prefix}/config/"
        
        config_content = None
        found_config_path = None
        
        try:
            # List objects in config directory to find matching execution ID
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=config_prefix,
                MaxKeys=1000  # Large limit to find all configs
            )
            
            if 'Contents' in response:
                # Look for config.yaml files in directories matching our execution ID
                for obj in response['Contents']:
                    key = obj['Key']
                    if key.endswith('config.yaml'):
                        # Check if this config path contains our execution ID
                        if f"-{execution_id}/" in key:
                            # Found the matching config!
                            try:
                                config_response = s3_client.get_object(Bucket=bucket, Key=key)
                                config_content = config_response['Body'].read().decode('utf-8')
                                found_config_path = key
                                print(f"Found config with execution ID {execution_id}: s3://{bucket}/{key}")
                                break
                            except Exception as e:
                                print(f" Could not download config from {key}: {e}")
                                continue
                
                if not config_content:
                    # Show what configs exist for debugging
                    config_dirs = set()
                    for obj in response['Contents']:
                        key = obj['Key']
                        if '/' in key and key.endswith('config.yaml'):
                            dir_path = '/'.join(key.split('/')[:-1])  # Remove filename
                            config_dirs.add(dir_path)
                    
                    if config_dirs:
                        return f"Config not found for execution ID '{execution_id}'. Available config directories:\n" \
                               f"   {chr(10).join(f'   • s3://{bucket}/{d}' for d in sorted(config_dirs))}"
                    else:
                        return f"No config directories found in s3://{bucket}/{config_prefix}"
            else:
                return f"No objects found in config directory: s3://{bucket}/{config_prefix}"
                
        except Exception as e:
            return f"Error searching S3 for config: {str(e)}"
        
        if config_content:
            return config_content
        else:
            return f"Config not found for execution ID '{execution_id}' in s3://{bucket}/{config_prefix}"
                   
    except Exception as e:
        return f"Error retrieving config: {str(e)}"


