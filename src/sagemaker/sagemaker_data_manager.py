"""
SageMaker Data Manager for YOLO Model Training.
Processes Ground Truth annotations and prepares YOLO dataset format.
Can run independently or be integrated into SageMaker pipelines later.
"""

import os
import json
import yaml
import argparse
from typing import Dict, Any, List, Tuple
from rich import print

import boto3
import s3fs
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils_config import load_config, load_ground_truth_config
from utils.utils_data import (
    parse_gt_output, 
    process_yolo_format, 
    save_yolo_annotations_to_s3,
)


class YOLODataManager:
    """
    Manages YOLO dataset preparation from Ground Truth annotations.
    Creates train/val splits and organizes data in YOLO format.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the YOLO Data Manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.gt_config = load_ground_truth_config()
        
        # Extract configuration values
        self.s3_bucket = self.gt_config.get('prod_s3_bucket')
        self.gt_bucket = self.gt_config.get('s3_bucket')
        self.job_id = self.gt_config.get('job_id')
        self.gt_job_name = self.gt_config.get('ground_truth_job_name')
        self.manifest_path = self.gt_config.get('output_manifest_file')
        self.images_dir = self.gt_config.get('images_dir')
        
        # Extract dataset configuration
        self.train_ratio = self.gt_config.get('train_ratio', 0.8)
        self.random_seed = self.gt_config.get('random_seed', 42)
        
        # Setup S3 paths for output
        self.output_prefix = f"labeling_task_{self.job_id}"
        self.s3_output_base = f"s3://{self.s3_bucket}/{self.output_prefix}"
        
        # Note: No local working directory needed - all processing done directly from S3
        
        print(f"Initialized YOLO Data Manager:")
        print(f"  Ground Truth Bucket: {self.gt_bucket}")
        print(f"  Output Bucket: {self.s3_bucket}")
        print(f"  Job ID: {self.job_id}")
        print(f"  Manifest: {self.manifest_path}")
        print(f"  Output Prefix: {self.output_prefix}")
    
    def get_manifest_info(self) -> Dict[str, Any]:
        """
        Get manifest information directly from S3 without downloading.
        
        Returns:
            Dictionary with manifest information including categories
        """
        print(f"Processing manifest directly from S3: {self.manifest_path}")
        
        try:
            # Use s3fs to read manifest directly from S3
            filesys = s3fs.S3FileSystem()
            
            categories = []
            total_annotations = 0
            unique_images = set()
            
            with filesys.open(self.manifest_path) as fin:
                for line_num, line in enumerate(fin.readlines()):
                    try:
                        record = json.loads(line)
                        if self.gt_job_name in record:
                            # Extract categories from metadata if available
                            metadata = record.get(f"{self.gt_job_name}-metadata", {})
                            class_map = metadata.get("class-map", {})
                            
                            if class_map and not categories:
                                # Convert class map to ordered list
                                categories = [""] * len(class_map)
                                for class_id, class_name in class_map.items():
                                    categories[int(class_id)] = class_name
                                print(f"Found {len(categories)} categories from manifest: {categories}")
                            
                            # Count annotations and images
                            if self.gt_job_name in record:
                                annotations = record[self.gt_job_name].get("annotations", [])
                                total_annotations += len(annotations)
                                
                                # Get image file path
                                image_file_path = record.get("source-ref", "")
                                if image_file_path:
                                    image_file_name = image_file_path.split("/")[-1]
                                    unique_images.add(image_file_name)
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed line {line_num + 1}: {e}")
                        continue
            
            if not categories:
                print("No categories found in manifest metadata")
                return {
                    'categories': [],
                    'total_annotations': total_annotations,
                    'unique_images': len(unique_images),
                    'manifest_path': self.manifest_path
                }
            
            print(f"Manifest processed successfully:")
            print(f"  - Categories: {len(categories)}")
            print(f"  - Total annotations: {total_annotations}")
            print(f"  - Unique images: {len(unique_images)}")
            
            return {
                'categories': categories,
                'total_annotations': total_annotations,
                'unique_images': len(unique_images),
                'manifest_path': self.manifest_path
            }
            
        except Exception as e:
            print(f"Error processing manifest: {e}")
            raise
    
    def extract_categories(self) -> List[str]:
        """
        Extract category names from the manifest file directly from S3.
        
        Returns:
            List of category names in proper order
        """
        print("Extracting category information from manifest...")
        
        try:
            # Get manifest info directly from S3
            manifest_info = self.get_manifest_info()
            categories = manifest_info.get('categories', [])
            
            if categories:
                print(f"Found {len(categories)} categories: {categories}")
                return categories
            
            # Fallback: try to use the existing get_categories function
            # Look for a labels file in the same S3 directory as manifest
            print("No categories found in manifest, trying to find labels file...")
            
            # Parse manifest S3 path to find labels file
            if self.manifest_path.startswith('s3://'):
                manifest_dir = '/'.join(self.manifest_path.split('/')[:-1])  # Remove filename
                labels_s3_path = f"{manifest_dir}/labels.json"
                
                try:
                    from utils.utils_data import get_categories
                    categories = get_categories(labels_s3_path)
                    print(f"Found {len(categories)} categories from labels file: {categories}")
                    return categories
                except Exception as e:
                    print(f"Could not read labels file {labels_s3_path}: {e}")
            
            raise ValueError("No category information found in manifest or labels file")
            
        except Exception as e:
            print(f"Error extracting categories: {e}")
            raise
    
    def process_annotations(self, categories: List[str]) -> pd.DataFrame:
        """
        Process Ground Truth annotations into YOLO format.
        
        Args:
            categories: List of category names
            
        Returns:
            DataFrame with YOLO format annotations
        """
        print("Processing Ground Truth annotations...")
        
        try:
            # Parse Ground Truth output using existing utils function
            df_bbox = parse_gt_output(self.gt_config)
            print(f"Parsed {len(df_bbox)} annotations")
            
            # Convert to YOLO format
            df_yolo = process_yolo_format(df_bbox, categories)
            print(f"Converted to YOLO format")
            
            return df_yolo
            
        except Exception as e:
            print(f"Error processing annotations: {e}")
            raise
    
    def create_train_val_split(self, df_annotations: pd.DataFrame, 
                              train_ratio: float = 0.8, 
                              random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation split of the dataset.
        
        Args:
            df_annotations: DataFrame with annotations
            train_ratio: Ratio of training data (default: 0.8)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df)
        """
        print(f"Creating train/val split (train: {train_ratio:.1%}, val: {1-train_ratio:.1%})...")
        
        # Get unique images
        unique_images = df_annotations['img_file'].unique()
        print(f"Total unique images: {len(unique_images)}")
        
        # Split images
        train_images, val_images = train_test_split(
            unique_images, 
            train_size=train_ratio, 
            random_state=random_seed,
            shuffle=True
        )
        
        # Split annotations based on images
        train_df = df_annotations[df_annotations['img_file'].isin(train_images)]
        val_df = df_annotations[df_annotations['img_file'].isin(val_images)]
        
        print(f"Train set: {len(train_df)} annotations from {len(train_images)} images")
        print(f"Val set: {len(val_df)} annotations from {len(val_images)} images")
        
        return train_df, val_df
    
    def copy_images_to_s3(self, train_images: List[str], val_images: List[str]) -> None:
        """
        Copy images to S3 in the proper directory structure.
        
        Args:
            train_images: List of training image filenames
            val_images: List of validation image filenames
        """
        print("Copying images to S3...")
        
        s3_client = boto3.client('s3')
        
        # Parse source images directory
        if self.images_dir.startswith('s3://'):
            source_bucket = self.images_dir.split('/')[2]
            source_prefix = '/'.join(self.images_dir.split('/')[3:])
        else:
            raise ValueError(f"Invalid images directory: {self.images_dir}")
        
        def copy_image_set(image_list: List[str], split_name: str):
            """Copy a set of images to the specified split directory."""
            target_prefix = f"{self.output_prefix}/{split_name}/images"
            
            for img_file in image_list:
                source_key = f"{source_prefix}/{img_file}"
                target_key = f"{target_prefix}/{img_file}"
                
                try:
                    # Copy image
                    copy_source = {'Bucket': source_bucket, 'Key': source_key}
                    s3_client.copy(copy_source, self.s3_bucket, target_key)
                    print(f"  {img_file} -> {split_name}/images/")
                except Exception as e:
                    print(f"  Error copying {img_file}: {e}")
        
        # Copy training images
        print(f"Copying {len(train_images)} training images...")
        copy_image_set(train_images, "train")
        
        # Copy validation images
        print(f"Copying {len(val_images)} validation images...")
        copy_image_set(val_images, "val")
        
        print("Image copying completed")
    
    def save_annotations_to_s3(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """
        Save YOLO format annotations to S3.
        
        Args:
            train_df: Training annotations DataFrame
            val_df: Validation annotations DataFrame
        """
        print("Saving YOLO annotations to S3...")
        
        try:
            # Save training annotations
            train_prefix = f"{self.output_prefix}/train/labels"
            save_yolo_annotations_to_s3(self.s3_bucket, train_prefix, train_df)
            print(f"Training annotations saved to: {train_prefix}")
            
            # Save validation annotations
            val_prefix = f"{self.output_prefix}/val/labels"
            save_yolo_annotations_to_s3(self.s3_bucket, val_prefix, val_df)
            print(f"Validation annotations saved to: {val_prefix}")
            
        except Exception as e:
            print(f"Error saving annotations: {e}")
            raise
    
    def create_data_yaml(self, categories: List[str]) -> str:
        """
        Create data.yaml file for YOLO training.
        
        Args:
            categories: List of category names
            
        Returns:
            Content of the data.yaml file
        """
        print("Creating data.yaml configuration...")
        
        data_yaml = {
            'path': f"s3://{self.s3_bucket}/{self.output_prefix}",
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(categories),
            'names': categories
        }
        
        yaml_content = yaml.dump(data_yaml, default_flow_style=False, sort_keys=False)
        
        # Save to S3
        s3_client = boto3.client('s3')
        yaml_key = f"{self.output_prefix}/data.yaml"
        
        try:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=yaml_key,
                Body=yaml_content,
                ContentType='text/yaml'
            )
            print(f"data.yaml saved to: s3://{self.s3_bucket}/{yaml_key}")
            return yaml_content
            
        except Exception as e:
            print(f"Error saving data.yaml: {e}")
            raise
    
    def copy_manifest_to_output(self) -> None:
        """Copy the original manifest file to the output directory for reference."""
        print("Copying manifest to output directory...")
        
        try:
            s3_client = boto3.client('s3')
            
            # Parse source manifest path
            if self.manifest_path.startswith('s3://'):
                source_bucket = self.manifest_path.split('/')[2]
                source_key = '/'.join(self.manifest_path.split('/')[3:])
            else:
                raise ValueError(f"Invalid manifest path: {self.manifest_path}")
            
            # Copy to output
            target_key = f"{self.output_prefix}/output.manifest"
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            s3_client.copy(copy_source, self.s3_bucket, target_key)
            
            print(f"Manifest copied to: s3://{self.s3_bucket}/{target_key}")
            
        except Exception as e:
            print(f"Error copying manifest: {e}")
            raise
    
    def process_dataset(self, train_ratio: float = None, random_seed: int = None) -> Dict[str, Any]:
        """
        Complete dataset processing workflow.
        
        Args:
            train_ratio: Ratio of training data (uses config value if None)
            random_seed: Random seed for reproducibility (uses config value if None)
            
        Returns:
            Dictionary with processing results
        """
        # Use config values if not provided
        if train_ratio is None:
            train_ratio = self.train_ratio
        if random_seed is None:
            random_seed = self.random_seed
            
        print("="*60)
        print("Starting YOLO Dataset Processing")
        print("="*60)
        print(f"Using train_ratio: {train_ratio}, random_seed: {random_seed}")
        
        try:
            # Step 1: Extract categories directly from S3 manifest
            categories = self.extract_categories()
            
            # Step 2: Process annotations
            df_annotations = self.process_annotations(categories)
            
            # Step 3: Create train/val split
            train_df, val_df = self.create_train_val_split(df_annotations, train_ratio, random_seed)
            
            # Step 4: Copy images to S3
            train_images = train_df['img_file'].unique().tolist()
            val_images = val_df['img_file'].unique().tolist()
            self.copy_images_to_s3(train_images, val_images)
            
            # Step 5: Save annotations
            self.save_annotations_to_s3(train_df, val_df)
            
            # Step 6: Create data.yaml
            data_yaml_content = self.create_data_yaml(categories)
            
            # Step 7: Copy manifest
            self.copy_manifest_to_output()
            
            print("="*60)
            print("YOLO Dataset Processing Completed Successfully!")
            print("="*60)
            
            # Return results
            result = {
                'status': 'success',
                'output_s3_path': f"s3://{self.s3_bucket}/{self.output_prefix}",
                'categories': categories,
                'total_images': len(train_images) + len(val_images),
                'train_images': len(train_images),
                'val_images': len(val_images),
                'total_annotations': len(df_annotations),
                'train_annotations': len(train_df),
                'val_annotations': len(val_df),
                'data_yaml': data_yaml_content
            }
            
            print("\nDataset Summary:")
            print(f"  Output Location: {result['output_s3_path']}")
            print(f"  Categories: {len(categories)} - {categories}")
            print(f"  Total Images: {result['total_images']}")
            print(f"  Train/Val Split: {result['train_images']}/{result['val_images']}")
            print(f"  Total Annotations: {result['total_annotations']}")
            print(f"  Train/Val Annotations: {result['train_annotations']}/{result['val_annotations']}")
            
            return result
            
        except Exception as e:
            print(f"Dataset processing failed: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the processed dataset.
        
        Returns:
            Dictionary with dataset information
        """
        print(f"Getting dataset info from: {self.s3_output_base}")
        
        try:
            from utils.utils_data import get_s3_dataset_info
            
            # Use the utility function
            info = get_s3_dataset_info(self.s3_bucket, self.output_prefix)
            
            if info['status'] == 'found':
                print("Dataset Information:")
                for key, value in info.items():
                    if key not in ['status', 'bucket', 'base_prefix']:
                        print(f"  {key}: {value}")
            
            return info
            
        except Exception as e:
            print(f"Error getting dataset info: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def process_for_sagemaker(self, train_ratio: float = None, random_seed: int = None) -> Dict[str, Any]:
        """
        Process dataset specifically for SageMaker processing jobs.
        This method handles SageMaker-specific output requirements.
        
        Args:
            train_ratio: Ratio of training data (uses config value if None)
            random_seed: Random seed for reproducibility (uses config value if None)
            
        Returns:
            Dictionary with processing results
        """
        print("="*60)
        print("Starting YOLO Dataset Processing for SageMaker")
        print("="*60)
        
        try:
            # Process dataset normally
            result = self.process_dataset(train_ratio, random_seed)
            
            # Add SageMaker-specific information
            result['sagemaker_ready'] = True
            result['processing_job_name'] = os.environ.get('SM_CURRENT_HOST', 'unknown')
            result['processing_job_arn'] = os.environ.get('SM_CURRENT_HOST_ARN', 'unknown')
            
            print("="*60)
            print("SageMaker Processing Completed Successfully!")
            print("="*60)
            
            return result
            
        except Exception as e:
            print(f"SageMaker processing failed: {e}")
            raise




def main():
    parser = argparse.ArgumentParser(description='YOLO Data Manager for SageMaker')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--action', type=str, choices=['process', 'info'], default='process',
                        help='Action to perform: process dataset or get info')
    args = parser.parse_args()
    
    try:
        # Initialize data manager
        data_manager = YOLODataManager(args.config)
        
        if args.action == 'process':
            result = data_manager.process_dataset()
            print(f"\nProcessing completed successfully!")
            
        elif args.action == 'info':
            info = data_manager.get_dataset_info()
            if info['status'] == 'found':
                print(f"\nDataset found and ready for training!")
            else:
                print(f"\nDataset status: {info['status']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
