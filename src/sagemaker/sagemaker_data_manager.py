"""
SageMaker data management utilities for plastic bag detection model.
Handles S3 uploads, data preparation, and S3 path management.
"""

import sagemaker
import boto3
import os
import sys
import argparse
from typing import Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_config,
    get_data_config,
    get_aws_config,
)



class SageMakerDataManager:
    """Handles S3 data operations for SageMaker training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SageMaker data manager with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML file
        """
        self.config = config or {}
        
        # Extract AWS configuration
        aws_config = get_aws_config(self.config)
        self.bucket = aws_config.get('bucket')
        self.prefix = aws_config.get('prefix', 'yolo-pipeline')
        region = aws_config.get('region')
        
        if not self.bucket:
            raise ValueError("Bucket must be specified in config.yaml")
            
        self.sess = sagemaker.Session()
        self.region = region or self.sess.boto_region_name
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # Ensure the destination bucket exists (create if missing)
        self._ensure_bucket_exists()
        
        # S3 paths
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
        # Parse S3 path
        if not s3_path.startswith('s3://'):
            return False
            
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        try:
            # List objects with the prefix to check if any exist
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
            return 'Contents' in response and len(response['Contents']) > 0
        except ClientError:
            return False
    
    def _get_local_file_size(self, file_path: str) -> int:
        """Get the size of a local file in bytes."""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0

    def _bucket_exists(self, bucket_name: str) -> bool:
        """Check if an S3 bucket exists and is accessible."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError as error:
            error_code = error.response.get('Error', {}).get('Code', '')
            # Treat Not Found/NoSuchBucket as non-existent; other errors bubble up
            if error_code in ('404', 'NoSuchBucket'):
                return False
            # In some cases, 301 (moved) indicates wrong region; treat as exists
            if error_code in ('301',):
                return True
            # 403 may indicate it exists but access denied; assume exists to avoid create conflict
            if error_code in ('403',):
                return True
            raise

    def _ensure_bucket_exists(self) -> None:
        """Create the S3 bucket if it does not already exist."""
        if self._bucket_exists(self.bucket):
            return
        print(f"Bucket '{self.bucket}' not found. Creating in region: {self.region}")
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={
                        'LocationConstraint': self.region
                    },
                )
        except ClientError as error:
            # If bucket already exists or owned by you, proceed; otherwise re-raise
            error_code = error.response.get('Error', {}).get('Code', '')
            if error_code not in ('BucketAlreadyOwnedByYou', 'BucketAlreadyExists'):
                raise
    
    def _get_s3_object_size(self, s3_path: str) -> int:
        """Get the size of an S3 object in bytes."""
        if not s3_path.startswith('s3://'):
            return 0
            
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response['ContentLength']
        except ClientError:
            return 0
    
    def check_data_in_s3(self) -> Tuple[bool, bool, Dict[str, str]]:
        """
        Check if training and validation directories already exist in S3.
        
        Returns:
            Tuple of (train_exists, validation_exists, s3_paths)
            where s3_paths is a dict with 'train' and 'validation' keys pointing to S3 prefixes
        """
        data_config = get_data_config(self.config)
        # Allow optional overrides via config under data: { s3_train_prefix, s3_val_prefix }
        expected_train_s3 = (
            data_config.get('s3_train_prefix')
            or f"s3://{self.bucket}/{self.prefix}/train/"
        )
        expected_validation_s3 = (
            data_config.get('s3_val_prefix')
            or data_config.get('s3_validation_prefix')
            or f"s3://{self.bucket}/{self.prefix}/val/"
        )
        
        # Check if any objects exist under the prefixes
        train_exists = self._s3_object_exists(expected_train_s3)
        validation_exists = self._s3_object_exists(expected_validation_s3)
        
        s3_paths = {
            'train': expected_train_s3 if train_exists else expected_train_s3,
            'validation': expected_validation_s3 if validation_exists else expected_validation_s3,
        }
        
        print(f"S3 data check results:")
        print(f"  Training prefix exists: {train_exists} ({expected_train_s3})")
        print(f"  Validation prefix exists: {validation_exists} ({expected_validation_s3})")
        
        return train_exists, validation_exists, s3_paths
    
    def upload_data_to_s3(self, force_upload: bool = False) -> Tuple[str, str]:
        """
        Upload training and validation directories to S3.
        
        Args:
            force_upload: If True, upload even if data already exists in S3
            
        Returns:
            Tuple of (train_s3_prefix, validation_s3_prefix)
        """
        # Check if data already exists in S3
        train_exists, validation_exists, existing_s3_paths = self.check_data_in_s3()
        
        # Determine local directories (allow overrides via config)
        data_config = get_data_config(self.config)
        train_dir = data_config.get('train_dir') or os.path.join('dataset', 'yolo_dataset', 'train')
        val_dir = data_config.get('val_dir') or data_config.get('validation_dir') or os.path.join('dataset', 'yolo_dataset', 'val')
        
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        # Expected S3 prefixes
        expected_train_s3 = (
            data_config.get('s3_train_prefix')
            or f"s3://{self.bucket}/{self.prefix}/train/"
        )
        expected_validation_s3 = (
            data_config.get('s3_val_prefix')
            or data_config.get('s3_validation_prefix')
            or f"s3://{self.bucket}/{self.prefix}/val/"
        )
        
        if not force_upload and train_exists and validation_exists:
            print("Data already exists in S3. Skipping upload.")
            self.s3_train_data = expected_train_s3
            self.s3_validation_data = expected_validation_s3
            return self.s3_train_data, self.s3_validation_data
        
        # Upload training directory if needed
        if force_upload or not train_exists:
            print(f"Uploading training directory: {train_dir}")
            train_channel = f"{self.prefix}/train"
            self.sess.upload_data(path=train_dir, bucket=self.bucket, key_prefix=train_channel)
            self.s3_train_data = expected_train_s3
            print(f"Training data uploaded to prefix: {self.s3_train_data}")
        else:
            self.s3_train_data = expected_train_s3
            print(f"Using existing training prefix: {self.s3_train_data}")
        
        # Upload validation directory if needed
        if force_upload or not validation_exists:
            print(f"Uploading validation directory: {val_dir}")
            validation_channel = f"{self.prefix}/val"
            self.sess.upload_data(path=val_dir, bucket=self.bucket, key_prefix=validation_channel)
            self.s3_validation_data = expected_validation_s3
            print(f"Validation data uploaded to prefix: {self.s3_validation_data}")
        else:
            self.s3_validation_data = expected_validation_s3
            print(f"Using existing validation prefix: {self.s3_validation_data}")
        
        # Print reminder about config
        if not data_config.get('s3_train_prefix') or not data_config.get('s3_val_prefix'):
            print(f"\n💡 Consider updating your config.yaml with these S3 prefixes under 'data':")
            print(f"   s3_train_prefix: \"{self.s3_train_data}\"")
            print(f"   s3_val_prefix: \"{self.s3_validation_data}\"")
            print(f"   train_dir: \"{os.path.abspath(train_dir)}\"")
            print(f"   val_dir: \"{os.path.abspath(val_dir)}\"")
        
        return self.s3_train_data, self.s3_validation_data
    
    def get_s3_data_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get S3 paths for training and validation data.
        
        Returns:
            Tuple of (train_s3_path, validation_s3_path)
        """
        return self.s3_train_data, self.s3_validation_data
    
    def set_s3_paths(self, train_s3_path: str, validation_s3_path: str):
        """
        Manually set S3 paths for training and validation data.
        Useful when data is already uploaded by external processes.
        
        Args:
            train_s3_path: S3 path to training data
            validation_s3_path: S3 path to validation data
        """
        self.s3_train_data = train_s3_path
        self.s3_validation_data = validation_s3_path
        
        print(f"Set S3 data paths:")
        print(f"  Train: {self.s3_train_data}")
        print(f"  Validation: {self.s3_validation_data}")


def main():
    parser = argparse.ArgumentParser(description="Upload YOLO data to S3")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force upload even if data exists in S3")
    parser.add_argument("--check-only", action="store_true", help="Only check if data exists, don't upload")
    
    args = parser.parse_args()
    
    # load config file
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
