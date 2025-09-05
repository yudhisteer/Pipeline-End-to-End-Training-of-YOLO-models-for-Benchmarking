import sys
import argparse
import boto3
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


from utils.utils_metrics import (
    get_training_job_details, 
    list_all_training_jobs_with_metrics, 
    list_specific_job_with_metrics,
    list_training_jobs, 
    get_training_job_config
)




def display_training_job_config(training_job_name: str):
    """
    Display the config.yaml used during a pipeline training job.
    Searches S3 for configs in the format: {prefix}/config/{timestamp}-{execution_id}/config.yaml
    
    Args:
        training_job_name: Name of the training job
    """
    console = Console()
    
    console.print(f"[bold cyan]🔧 Retrieving config for training job:[/bold cyan] [yellow]{training_job_name}[/yellow]\n")
    
    # Get config content
    config_content = get_training_job_config(training_job_name)
    
    if config_content.startswith("❌"):
        console.print(f"[bold red]{config_content}[/bold red]")
        return
    
    # Try to parse and display as YAML
    try:
        # Parse YAML to validate and format
        config_data = yaml.safe_load(config_content)
        
        # Display config in a nice panel
        console.print(Panel(
            f"[bold green]Pipeline Training Config.yaml[/bold green]\n"
            f"[dim]Training Job: {training_job_name}[/dim]",
            title="[bold blue]📋 Pipeline Training Configuration[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        ))
        
        # Display as formatted YAML
        syntax = Syntax(config_content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)
        
    except yaml.YAMLError as e:
        # If YAML parsing fails, display as plain text
        console.print(f"[bold yellow] Config is not valid YAML:[/bold yellow] {e}")
        console.print(f"[dim]Displaying as plain text:[/dim]")
        console.print(config_content)


def display_training_job_metrics(
    training_job_name: str = None, 
    show_metrics: bool = False, 
    max_results: int = 10,
    show_config: bool = False
    ) -> None:
    """
    Display training job information with optional metrics and config.
    
    Args:
        training_job_name: Specific job name to display metrics for
        show_metrics: Whether to show detailed metrics for all jobs
        max_results: Maximum number of training jobs to display
        show_config: Whether to show the config.yaml used during training
    """
    console = Console()
    
    if training_job_name:
        # Display metrics for specific job only if requested
        console.print(f"[bold cyan] Displaying information for:[/bold cyan] [yellow]{training_job_name}[/yellow]\n")
        job_details = get_training_job_details(training_job_name)
        if job_details:
            # Only show metrics if explicitly requested
            if show_metrics:
                list_specific_job_with_metrics(job_details)
            
            # Display config if requested
            if show_config:
                if show_metrics:
                    console.print("\n" + "="*80)
                display_training_job_config(training_job_name)
    else:
        # Display all jobs - with or without metrics based on flag
        if show_metrics:
            console.print(f"[bold cyan]📊 Displaying top {max_results} recent training jobs with metrics[/bold cyan]\n")
            list_all_training_jobs_with_metrics(max_results=max_results)
        else:
            console.print(f"[bold cyan]📋 Displaying top {max_results} recent training jobs[/bold cyan]\n")
            list_training_jobs(max_results=max_results)



def main():
    """Main function to handle command line arguments and display training job metrics."""
    parser = argparse.ArgumentParser(
        description="SageMaker Training Job Metrics Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Display basic job list (status, name, creation time) - default 10 jobs
        python src/sagemaker/sagemaker_metrics.py
        
        # Display top 5 jobs with detailed metrics
        python src/sagemaker/sagemaker_metrics.py --metrics --top_n 5
        
        # Display all jobs with detailed metrics (default 10)
        python src/sagemaker/sagemaker_metrics.py --metrics
        
        # Display top 3 basic job information
        python src/sagemaker/sagemaker_metrics.py --top_n 3
        
        # Display metrics for specific training job
        python src/sagemaker/sagemaker_metrics.py job-name-here
        
        # Display config.yaml used during pipeline training job
        python src/sagemaker/sagemaker_metrics.py job-name-here --config
        
        # Display both metrics and config for specific training job
        python src/sagemaker/sagemaker_metrics.py job-name-here --metrics --config
        """
    )
    
    parser.add_argument(
        'job_name',
        nargs='?',
        help='Specific training job name to display metrics for'
    )
    
    parser.add_argument(
        '--metrics',
        action='store_true',
        help='Show detailed metrics for all training jobs (default: basic info only)'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show the config.yaml used during pipeline training (requires job_name, searches S3 for {prefix}/config/{timestamp}-{execution_id}/config.yaml)'
    )
    
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='Number of most recent training jobs to display (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.config and not args.job_name:
        parser.error("--config requires a job_name to be specified")
    
    console = Console()
    
    console.print(Panel(
        "[bold blue]SageMaker Training Job Metrics Extractor[/bold blue]\n"
        "[dim]Fetches and displays training metrics from SageMaker jobs[/dim]",
        title="[bold green]🚀 YOLO Training Metrics[/bold green]",
        border_style="green",
        padding=(1, 2),
        width=180
    ))
    
    if args.job_name:
        console.print(f"[bold cyan]Looking for job:[/bold cyan] [yellow]{args.job_name}[/yellow]\n")
        display_training_job_metrics(
            args.job_name, 
            show_metrics=args.metrics, 
            show_config=args.config
        )
    else:
        if args.metrics:
            console.print(f"[bold yellow]💡 Mode:[/bold yellow] Detailed metrics for top {args.top_n} jobs")
        else:
            console.print(f"[bold yellow]💡 Mode:[/bold yellow] Basic job information for top {args.top_n} \
                jobs (use --metrics for detailed view)")
        console.print()
        display_training_job_metrics(show_metrics=args.metrics, max_results=args.top_n)


if __name__ == "__main__":
    main()