


```bash

# Show help message with all available options
python src/sagemaker/sagemaker_data_manager.py --help

# Check if data exists in S3 then upload (uses default config.yaml)
python src/sagemaker/sagemaker_data_manager.py

# Check if data exists in S3, then force upload (overwrites existing data)
python src/sagemaker/sagemaker_data_manager.py --config config.yaml --force --check-only

# Force upload data to S3 (overwrites existing data)
python src/sagemaker/sagemaker_data_manager.py --config config.yaml --force

# Only check if data exists in S3 (no upload)
python src/sagemaker/sagemaker_data_manager.py --config config.yaml --check-only

# Check if data exists, upload only if missing (default behavior)
python src/sagemaker/sagemaker_data_manager.py --config config.yaml

```