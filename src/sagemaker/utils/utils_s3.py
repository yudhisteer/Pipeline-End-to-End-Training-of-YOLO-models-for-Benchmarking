import boto3
import os
from typing import Tuple, List
from urllib.parse import urlparse
from pathlib import Path



def object_exists(s3_client: boto3.client, bucket: str, key: str) -> bool:
    """
    Check if an S3 object exists.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        True if object exists, False otherwise
    """
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False



def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse S3 URI into bucket and key components.
    
    Args:
        s3_uri: S3 URI in format s3://bucket/key
        
    Returns:
        Tuple of (bucket, key)
    """
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")


# ------------------------------------------------------------------------------------------------ #
# function to upload images in a folder to s3
# ------------------------------------------------------------------------------------------------ #
def upload_images_to_s3(
    s3_client: boto3.client,
    local_folder: str,
    bucket: str,
    s3_prefix: str = "bounding_box/images/"
    ) -> None:
    """
    Function that replicates your exact AWS CLI command:
    aws s3 sync . s3://ground-truth-data-labeling/bounding_box/images/ --exclude "*" --include "*.jpg"
    
    Args:
        s3_client: Boto3 S3 client
        local_folder: Local folder path (use "." for current directory)
        bucket: S3 bucket name (e.g., "ground-truth-data-labeling")
        s3_prefix: S3 key prefix (default: "bounding_box/images/")
        
    Returns:
        None
    """

    def _check_images_to_upload_to_s3(
        s3_client: boto3.client, 
        local_folder: str, 
        bucket: str, 
        s3_prefix: str = "",
        file_extensions: List[str] = None,
        exclude_patterns: List[str] = None
        ) -> None:
        """
        Upload images from a local folder to S3 bucket, replicating aws s3 sync behavior.
        
        Args:
            s3_client: Boto3 S3 client
            local_folder: Local folder path containing images
            bucket: S3 bucket name
            s3_prefix: S3 key prefix (e.g., "bounding_box/images/")
            file_extensions: List of file extensions to include (default: [".jpg"])
            exclude_patterns: List of patterns to exclude (default: ["*"])
            
        Returns:
            None
        """
        if file_extensions is None:
            file_extensions = [".jpg"]
        
        if exclude_patterns is None:
            exclude_patterns = ["*"]
        
        # Ensure local folder exists
        if not os.path.exists(local_folder):
            raise ValueError(f"Local folder does not exist: {local_folder}")
        
        # Get all files in the local folder
        local_path = Path(local_folder)
        
        # Find all files with specified extensions
        matching_files = []
        for ext in file_extensions:
            matching_files.extend(local_path.glob(f"*{ext}"))
        
        print(f"Found {len(matching_files)} files to upload")
        
        # Upload each matching file
        for file_path in matching_files:
            # Create S3 key by combining prefix with filename
            s3_key = os.path.join(s3_prefix, file_path.name).replace("\\", "/")
            
            try:
                print(f"Uploading {file_path} to s3://{bucket}/{s3_key}")
                s3_client.upload_file(
                    str(file_path),
                    bucket,
                    s3_key
                )
                print(f"Successfully uploaded {file_path.name}")
            except Exception as e:
                print(f"Failed to upload {file_path.name}: {str(e)}")

    # check if images exist in s3
    _check_images_to_upload_to_s3(
        s3_client=s3_client,
        local_folder=local_folder,
        bucket=bucket,
        s3_prefix=s3_prefix,
        file_extensions=[".jpg"],
        exclude_patterns=["*"]
    )




# ------------------------------------------------------------------------------------------------ #
# Download annotated image and their corresponding yolo annotations
# ------------------------------------------------------------------------------------------------ #

def download_yolo_annotations(
    s3_client: boto3.client,
    bucket: str,
    base_filenames: str | List[str],
    local_dir: str = ".",
    annotation_prefix: str = "bounding_box/yolo_annot_files/",
    image_prefix: str = "bounding_box/images/"
) -> List[Tuple[str, str]]:
    """
    Download YOLO annotation files and their corresponding images from S3.
    Handles both single and multiple downloads automatically.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        base_filenames: Single filename (str) or list of filenames (List[str]) without extensions
        local_dir: Local directory to download files to
        annotation_prefix: S3 prefix for annotation files
        image_prefix: S3 prefix for image files
        
    Returns:
        List of tuples containing (annotation_file_path, image_file_path) for each downloaded pair
        
    Raises:
        ValueError: If local directory doesn't exist
        Exception: If S3 download fails
    """
    
    def _download_yolo_annotation_and_image(
        s3_client: boto3.client,
        bucket: str,
        base_filename: str,
        local_dir: str = ".",
        annotation_prefix: str = "bounding_box/yolo_annot_files/",
        image_prefix: str = "bounding_box/images/"
    ) -> Tuple[str, str]:
        """
        Download a YOLO annotation file and its corresponding image from S3.
        
        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name (e.g., "ground-truth-data-labeling")
            base_filename: Base filename without extension (e.g., "IMG_20200816_205004")
            local_dir: Local directory to download files to (default: current directory)
            annotation_prefix: S3 prefix for annotation files (default: "bounding_box/yolo_annot_files/")
            image_prefix: S3 prefix for image files (default: "bounding_box/images/")
            
        Returns:
            Tuple of (annotation_file_path, image_file_path) - local paths to downloaded files
            
        Raises:
            ValueError: If local directory doesn't exist
            Exception: If S3 download fails
        """
        # Ensure local directory exists
        if not os.path.exists(local_dir):
            raise ValueError(f"Local directory does not exist: {local_dir}")
        
        # Construct S3 keys and local file paths
        annotation_key = f"{annotation_prefix}{base_filename}.txt"
        image_key = f"{image_prefix}{base_filename}.jpg"
        
        annotation_local_path = os.path.join(local_dir, f"{base_filename}.txt")
        image_local_path = os.path.join(local_dir, f"{base_filename}.jpg")
        
        # Check if files exist in S3 before downloading
        if not object_exists(s3_client, bucket, annotation_key):
            raise ValueError(f"Annotation file does not exist in S3: s3://{bucket}/{annotation_key}")
        
        if not object_exists(s3_client, bucket, image_key):
            raise ValueError(f"Image file does not exist in S3: s3://{bucket}/{image_key}")
        
        try:
            # Download annotation file
            print(f"Downloading annotation file: s3://{bucket}/{annotation_key}")
            s3_client.download_file(bucket, annotation_key, annotation_local_path)
            print(f"Successfully downloaded annotation file to: {annotation_local_path}")
            
            # Download image file
            print(f"Downloading image file: s3://{bucket}/{image_key}")
            s3_client.download_file(bucket, image_key, image_local_path)
            print(f"Successfully downloaded image file to: {image_local_path}")
            
            return annotation_local_path, image_local_path
            
        except Exception as e:
            # Clean up any partially downloaded files
            for file_path in [annotation_local_path, image_local_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            raise Exception(f"Failed to download files: {str(e)}")
    
    if not os.path.exists(local_dir):
        raise ValueError(f"Local directory does not exist: {local_dir}")
    
    # Convert single filename to list for consistent processing
    if isinstance(base_filenames, str):
        base_filenames = [base_filenames]
    
    downloaded_files = []
    
    for base_filename in base_filenames:
        try:
            annotation_path, image_path = _download_yolo_annotation_and_image(
                s3_client=s3_client,
                bucket=bucket,
                base_filename=base_filename,
                local_dir=local_dir,
                annotation_prefix=annotation_prefix,
                image_prefix=image_prefix
            )
            downloaded_files.append((annotation_path, image_path))
        except Exception as e:
            print(f"Failed to download files for {base_filename}: {str(e)}")
            # Continue with other files instead of failing completely
            continue
    
    return downloaded_files



