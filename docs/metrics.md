


```bash
# Default behavior - show top 10 basic job info
python src/sagemaker/sagemaker_metrics.py

# Show top 5 jobs with detailed metrics  
python src/sagemaker/sagemaker_metrics.py --metrics --top_n 5

# Show top 3 basic job information
python src/sagemaker/sagemaker_metrics.py --top_n 3

# Show detailed metrics for default 10 jobs
python src/sagemaker/sagemaker_metrics.py --metrics

# Show metrics for a specific job
python src/sagemaker/sagemaker_metrics.py <job-name>
```



```bash
# usage
python src/sagemaker/sagemaker_metrics.py --help

# output:
usage: sagemaker_metrics.py [-h] [--metrics] [--top_n TOP_N] [job_name]

SageMaker Training Job Metrics Extractor

positional arguments:
  job_name       Specific training job name to display metrics for

options:
  -h, --help     show this help message and exit
  --metrics      Show detailed metrics for all training jobs (default: basic info only)
  --top_n TOP_N  Number of most recent training jobs to display (default: 10)

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
```



```bash
# run
python src/sagemaker/sagemaker_metrics.py

# output:
╭─────────────────────────────────────────── 🚀 YOLO Training Metrics ────────────────────────────────────────────╮
│                                                                                                                 │
│  SageMaker Training Job Metrics Extractor                                                                       │
│  Fetches and displays training metrics from SageMaker jobs                                                      │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
💡 Mode: Basic job information (use --metrics for detailed view)

📋 Displaying all recent training jobs


📋 Recent Training Jobs
⠴ Fetching training jobs...
                                              Recent Training Jobs
╭─────┬────────────────────────────────────────────────────────┬──────────────┬──────────────────┬──────────────╮
│ #   │ Job Name                                               │    Status    │ Creation Time    │ Duration (s) │
├─────┼────────────────────────────────────────────────────────┼──────────────┼──────────────────┼──────────────┤
│ 1   │ pipelines-d7r1ut4ei4h1-YOLOTrainingStep-Obj-lvkXGFIrlt │  Completed   │ 08-14 13:37      │          306 │
│ 2   │ pipelines-02sxsuysvct3-YOLOTrainingStep-lhZGWHuAgK     │  Completed   │ 08-14 12:29      │          315 │
│ 3   │ pipelines-yvesj8h8mbee-YOLOTrainingStep-lmLaL0y7i1     │  Completed   │ 08-14 12:21      │          316 │
│ 4   │ pipelines-2var0wrhfjvj-YOLOTrainingStep-J8bYsmg7IQ     │  Completed   │ 08-14 12:08      │          306 │
│ 5   │ pipelines-8p59s3hpsrzz-YOLOTrainingStep-6TpKabCaUG     │  Completed   │ 08-14 10:13      │          317 │
│ 6   │ pipelines-kosejghapxrc-YOLOTrainingStep-yiDdc0iaNp     │  Completed   │ 08-14 09:47      │          306 │
│ 7   │ pipelines-t0u47ewzhapi-YOLOTrainingStep-9dlURbrX2Y     │  Completed   │ 08-14 00:24      │          306 │
│ 8   │ pipelines-fnmjm6a1g9ox-YOLOTrainingStep-t79fabLPiW     │    Failed    │ 08-14 00:15      │          N/A │
│ 9   │ pipelines-shaskctivjwl-YOLOTrainingStep-sSQZaAnBRh     │    Failed    │ 08-14 00:06      │          N/A │
│ 10  │ pipelines-q67p4rblut08-YOLOTrainingStep-nk9CKPaemi     │    Failed    │ 08-14 00:00      │          N/A │
╰─────┴────────────────────────────────────────────────────────┴──────────────┴──────────────────┴──────────────╯
```





```bash
# run
python src/sagemaker/sagemaker_metrics.py --metrics --top_n 3

# output:
╭───────────────────────────────────────────────────────────── 🚀 YOLO Training Metrics ─────────────────────────────────────────────────────────────╮
│                                                                                                                                                    │
│  SageMaker Training Job Metrics Extractor                                                                                                          │
│  Fetches and displays training metrics from SageMaker jobs                                                                                         │
│                                                                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
💡 Mode: Detailed metrics for top 3 jobs

📊 Displaying top 3 recent training jobs with metrics


📋 Recent Training Jobs with Metrics
⠇ Fetching training jobs and metrics...
  📦 Downloaded 69,682,240 bytes
  ✅ Found results file: train/results.csv

  🏆 Best Epoch (highest mAP50):                                     1
⠦ 📥 Downloading from s3://cyudhist-pipeline-yolo-503561429929/yolo-pipeline/models/20250814-133723/pipelines-d7r1ut4ei4h1-YOLOTrainingStep-Obj-lvkXGFIrlt/output/model.tar.gz
⠦ 🔍 Extracting and searching files...
  📦 Downloaded 69,681,584 bytes
  ✅ Found results file: train/results.csv

  🏆 Best Epoch (highest mAP50):                                     1
⠋ 📥 Downloading from s3://cyudhist-pipeline-yolo-503561429929/yolo-pipeline/models/20250814-122931/pipelines-02sxsuysvct3-YOLOTrainingStep-lhZGWHuAgK/output/model.tar.gz
⠋ 🔍 Extracting and searching files...
  📦 Downloaded 69,776,947 bytes
  ✅ Found results file: train/results.csv

  🏆 Best Epoch (highest mAP50):                                     2
⠇ 📥 Downloading from s3://cyudhist-pipeline-yolo-503561429929/yolo-pipeline/models/20250814-122124/pipelines-yvesj8h8mbee-YOLOTrainingStep-lmLaL0y7i1/output/model.tar.gz
⠇ 🔍 Extracting and searching files...
                                                                              Recent Training Jobs with S3 Metrics
╭─────┬────────────────────────────────────────────┬────────────┬──────────────────┬────────────┬────────────┬──────────────┬────────────┬────────────┬──────────────┬────────────╮
│     │                                            │            │                  │            │       Best │         Best │            │       Last │         Last │   Duration │
│ #   │ Job Name                                   │   Status   │ Creation Time    │ Best mAP50 │     Recall │    Precision │ Last mAP50 │     Recall │    Precision │        (s) │
├─────┼────────────────────────────────────────────┼────────────┼──────────────────┼────────────┼────────────┼──────────────┼────────────┼────────────┼──────────────┼────────────┤
│ 1   │ pipelines-d7r1ut4ei4h1-YOLOTrainingStep…   │ Completed  │ 08-14 13:37      │     0.1477 │     0.2041 │       0.4433 │     0.1477 │     0.2041 │       0.4433 │        306 │
│ 2   │ pipelines-02sxsuysvct3-YOLOTrainingStep…   │ Completed  │ 08-14 12:29      │     0.1477 │     0.2041 │       0.4433 │     0.1477 │     0.2041 │       0.4433 │        315 │
│ 3   │ pipelines-yvesj8h8mbee-YOLOTrainingStep…   │ Completed  │ 08-14 12:21      │     0.1650 │     0.1837 │       0.4882 │     0.0071 │     0.0816 │       0.0170 │        316 │
╰─────┴────────────────────────────────────────────┴────────────┴──────────────────┴────────────┴────────────┴──────────────┴────────────┴────────────┴──────────────┴────────────╯
```


```bash
# run
python src/sagemaker/sagemaker_metrics.py pipelines-yvesj8h8mbee-YOLOTrainingStep-lmLaL0y7i1

# output:
╭──────────────────────────────────────────────────────────────────────────── 🚀 YOLO Training Metrics ────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                  │
│  SageMaker Training Job Metrics Extractor                                                                                                                                        │
│  Fetches and displays training metrics from SageMaker jobs                                                                                                                       │
│                                                                                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
🔍 Looking for job: pipelines-yvesj8h8mbee-YOLOTrainingStep-lmLaL0y7i1

🎯 Displaying metrics for: pipelines-yvesj8h8mbee-YOLOTrainingStep-lmLaL0y7i1


🔍 Attempting to extract metrics from S3 artifacts...
  📦 Downloaded 69,776,947 bytes
  ✅ Found results file: train/results.csv

  🏆 Best Epoch (highest mAP50):                                     2
⠧ 📥 Downloading from s3://cyudhist-pipeline-yolo-503561429929/yolo-pipeline/models/20250814-122124/pipelines-yvesj8h8mbee-YOLOTrainingStep-lmLaL0y7i1/output/model.tar.gz
⠧ 🔍 Extracting and searching files...

📊 Extracted Training Metrics
    Final Epoch Metrics
╭────────────────┬────────╮
│ Metric         │  Value │
├────────────────┼────────┤
│ epoch          │      5 │
│ precision      │ 0.0170 │
│ recall         │ 0.0816 │
│ mAP50          │ 0.0071 │
│ mAP50-95       │ 0.0024 │
│ train_box_loss │ 2.5900 │
│ val_box_loss   │ 2.8432 │
╰────────────────┴────────╯
   🏆 Best Epoch Metrics
      (Highest mAP50)
╭────────────────┬────────╮
│ Metric         │  Value │
├────────────────┼────────┤
│ epoch          │      2 │
│ precision      │ 0.4882 │
│ recall         │ 0.1837 │
│ mAP50          │ 0.1650 │
│ mAP50-95       │ 0.1132 │
│ train_box_loss │ 1.9606 │
│ val_box_loss   │ 1.1278 │
╰────────────────┴────────╯
```