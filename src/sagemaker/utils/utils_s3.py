import boto3
from typing import Tuple
from urllib.parse import urlparse



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


