"""
SageMaker data management utilities for YOLO model.
Handles S3 uploads, data preparation, and S3 path management.
"""

import sagemaker
import boto3
import os
import argparse
from typing import Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError


from utils.utils_config import (
    load_config,
    get_data_config,
    get_aws_config,
)



class SageMakerDataManager:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SageMaker data manager with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML file
        """
        self.config = config or {}
        
        # extract aws configuration
        aws_config = get_aws_config(self.config)
        self.bucket = aws_config.get('bucket')
        self.prefix = aws_config.get('prefix', 'yolo-pipeline')
        region = aws_config.get('region')
        
        if not self.bucket:
            raise ValueError("Bucket must be specified in config.yaml")
            
        self.sess = sagemaker.Session()
        self.region = region or self.sess.boto_region_name
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # create bucket if it does not exist
        self._ensure_bucket_exists()
        
        self.s3_train_data = None
        self.s3_validation_data = None
        
        print(f"Initialized SageMaker data manager for bucket: {self.bucket}, prefix: {self.prefix}")
    
    def _s3_object_exists(self, s3_path: str) -> bool:
        """
        Check if an S3 object exists.
        
        Args:
            s3_path: Full S3 path (s3://bucket/key)
            
        Returns:
            True if object exists, False otherwise
        """
        # extract bucket and key from s3 path
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        try:
            # list objects with the prefix to check if any exist
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
            return 'Contents' in response and len(response['Contents']) > 0
        except ClientError:
            return False

    def _bucket_exists(self, bucket_name: str) -> bool:
        """Check if an S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as error:
            error_code = error.response.get('Error', {}).get('Code', '')
            # treat Not Found/NoSuchBucket as non-existent
            if error_code in ('404', 'NoSuchBucket'):
                return False
            # treat 301 (moved) as exists
            if error_code in ('301',):
                return True
            # treat 403 (access denied) as exists: TODO: check if this is correct
            if error_code in ('403',):
                return True
            raise

    def _ensure_bucket_exists(self) -> None:
        """Create the S3 bucket if it does not already exist."""
        if self._bucket_exists(self.bucket):
            return
        print(f"Bucket '{self.bucket}' not found. Creating in region: {self.region}")
        try:
            if self.region == 'us-east-1': # TODO: should this be in config?
                self.s3_client.create_bucket(Bucket=self.bucket)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.region
                    },
                )
        except ClientError as error:
            # if bucket already exists or owned by you, proceed; otherwise re-raise
            error_code = error.response.get('Error', {}).get('Code', '')
            if error_code not in ('BucketAlreadyOwnedByYou', 'BucketAlreadyExists'):
                raise
    
    def check_data_in_s3(self) -> Tuple[bool, bool, Dict[str, str]]:
        """
        Check if training and validation directories already exist in S3.
        
        Returns:
            Tuple of (train_exists, validation_exists, s3_paths)
            where s3_paths is a dict with 'train' and 'validation' keys pointing to S3 prefixes
        """
        data_config = get_data_config(self.config)
        # Check for complete YOLO dataset
        expected_dataset_s3 = (
            data_config.get('s3_dataset_prefix')
            or f"s3://{self.bucket}/{self.prefix}/yolo_dataset/"
        )
        

        dataset_exists = self._s3_object_exists(os.path.join(expected_dataset_s3, 'data.yaml'))
        
        if dataset_exists:
            train_exists = True
            validation_exists = True
            print(f"S3 data check results:")
            print(f"  Complete YOLO dataset exists: {dataset_exists} ({expected_dataset_s3})")
        else:
            # No dataset found
            train_exists = False
            validation_exists = False
            print(f"S3 data check results:")
            print(f"  Complete YOLO dataset not found at: {expected_dataset_s3}")
        
        s3_paths = {
            'train': expected_dataset_s3,
            'validation': expected_dataset_s3,
        }
        
        return train_exists, validation_exists, s3_paths
    
    def upload_data_to_s3(self, force_upload: bool = False) -> Tuple[str, str]:
        """
        Upload training and validation directories to S3.
        
        Args:
            force_upload: If True, upload even if data already exists in S3
            
        Returns:
            Tuple of (train_s3_prefix, validation_s3_prefix)
        """
        # check if data already exists in S3
        train_exists, validation_exists, existing_s3_paths = self.check_data_in_s3()
        data_config = get_data_config(self.config)
        
        # For YOLO, we need to upload the entire dataset directory that contains data.yaml
        dataset_dir = data_config.get('dataset_dir') or os.path.join('dataset', 'yolo_dataset')
        train_dir = data_config.get('train_dir') or os.path.join(dataset_dir, 'train')
        val_dir = data_config.get('val_dir') or data_config.get('validation_dir') or os.path.join(dataset_dir, 'val')
        data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
        
        if not os.path.isdir(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        if not os.path.isfile(data_yaml_path):
            raise FileNotFoundError(f"YOLO data.yaml file not found: {data_yaml_path}")
        
        # expected s3 prefix for complete dataset
        expected_dataset_s3 = (
            data_config.get('s3_dataset_prefix')
            or f"s3://{self.bucket}/{self.prefix}/yolo_dataset/"
        )
        
        # Check if complete dataset exists
        dataset_exists = self._s3_object_exists(os.path.join(expected_dataset_s3, 'data.yaml'))
        
        if not force_upload and dataset_exists:
            print("Complete YOLO dataset already exists in S3. Skipping upload.")
            self.s3_train_data = expected_dataset_s3
            self.s3_validation_data = expected_dataset_s3
            return self.s3_train_data, self.s3_validation_data
        
        # Upload the complete dataset directory (includes data.yaml, train/, val/)
        print(f"Uploading complete YOLO dataset directory: {dataset_dir}")
        dataset_channel = f"{self.prefix}/yolo_dataset"
        self.sess.upload_data(path=dataset_dir, bucket=self.bucket, key_prefix=dataset_channel)
        self.s3_train_data = expected_dataset_s3
        self.s3_validation_data = expected_dataset_s3
        print(f"Complete dataset uploaded to prefix: {expected_dataset_s3}")
        print(f"  - Includes: data.yaml, train/images, train/labels, val/images, val/labels")
        
        if not data_config.get('s3_dataset_prefix'):
            print(f"\n Consider updating your config.yaml with this S3 prefix under 'data':")
            print(f"   s3_dataset_prefix: \"{expected_dataset_s3}\"")
            print(f"   dataset_dir: \"{os.path.abspath(dataset_dir)}\"")
        
        return self.s3_train_data, self.s3_validation_data
    


def main():
    parser = argparse.ArgumentParser(description="Upload YOLO data to S3")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force upload even if data exists in S3")
    parser.add_argument("--check-only", action="store_true", help="Only check if data exists, don't upload")
    
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # create data manager
    data_manager = SageMakerDataManager(config)
    
    # check if data exists in S3
    if args.check_only:
        print("Checking data in S3...")
        train_exists, validation_exists, s3_paths = data_manager.check_data_in_s3()
        
        if train_exists and validation_exists:
            print("All data exists in S3")
            print(f"Train: {s3_paths['train']}")
            print(f"Validation: {s3_paths['validation']}")
        else:
            print("Some data missing from S3")
            if not train_exists:
                print("Missing: Training data")   
            if not validation_exists:
                print("Missing: Validation data")
    else:
        # upload data to S3
        print("Uploading data to S3...")
        train_s3, validation_s3 = data_manager.upload_data_to_s3(force_upload=args.force)
        print(f"Process complete!")
        print(f"Train: {train_s3}")
        print(f"Validation: {validation_s3}")


if __name__ == "__main__":
    main()

    # Example usage:
    """
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
    """
