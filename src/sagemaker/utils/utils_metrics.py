import io
import os
import tarfile
import tempfile

import boto3
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


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
            f"[bold red]❌ Error:[/bold red] Training job '[yellow]{job_name}[/yellow]' not found: {e}"
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

    console.print("\n[bold blue]📋 Recent Training Jobs[/bold blue]", style="bold")

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
        "\n[bold blue]📋 Recent Training Jobs with Metrics[/bold blue]", style="bold"
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
                "[red]❌ No job details provided[/red]",
                title="[red]Error[/red]",
                border_style="red",
            )
        )
        return

    console.print(
        f"\n[bold yellow]🔍 Attempting to extract metrics from S3 artifacts...[/bold yellow]"
    )
    try:
        s3_uri = job_details["ModelArtifacts"]["S3ModelArtifacts"]
        metrics = extract_metrics_from_s3(s3_uri, console)

        if metrics:
            console.print(f"\n[bold green]📊 Extracted Training Metrics[/bold green]")

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
                    title="🏆 Best Epoch Metrics (Highest mAP50)", box=box.ROUNDED
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
                    f"[yellow]⚠️ Could not extract metrics from S3 artifacts[/yellow]\n"
                    f"[dim]📋 Manual extraction required from:[/dim]\n[blue]{s3_uri}[/blue]",
                    title="[red]Extraction Failed[/red]",
                    border_style="red",
                )
            )
    except Exception as e:
        console.print(
            Panel(
                f"[red]Error extracting metrics: {e}[/red]\n"
                f"[dim]Check S3 artifacts manually:[/dim]\n[blue]{job_details['ModelArtifacts']['S3ModelArtifacts']}[/blue]",
                title="[red]❌ Extraction Error[/red]",
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
                f"📥 Downloading from s3://{bucket}/{key}", total=None
            )

            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", delete=False
            ) as tmp_file:
                s3.download_file(bucket, key, tmp_file.name)
                progress.update(download_task, completed=True)
                console.print(
                    f"  [green]📦 Downloaded {os.path.getsize(tmp_file.name):,} bytes[/green]"
                )

                extract_task = progress.add_task(
                    "🔍 Extracting and searching files...", total=None
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
                        file_type = "📁 Dir" if member.isdir() else "📄 File"
                        archive_table.add_row(member.name, file_type)

                        # look for results.csv which contains the actual training metrics
                        if member.name.endswith("results.csv"):
                            results_file = member

                    # console.print(archive_table)

                    if results_file:
                        console.print(
                            f"  [bold green]✅ Found results file:[/bold green] [yellow]{results_file.name}[/yellow]"
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
                                f"\n  [bold gold1]🏆 Best Epoch (highest mAP50): \
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
