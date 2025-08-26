import matplotlib.pyplot as plt
import os
import random
import tempfile
import boto3
import base64       
from typing import List, Tuple

import matplotlib.image as mpimg


def list_local_models():
    """List all locally available models organized by training job."""
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print("No local models found.")
        return
    
    print("📁 Locally Available Models:")
    print("=" * 50)
    
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if os.path.isdir(item_path):
            # look for ONNX files in dir.
            onnx_files = []
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.endswith(".onnx"):
                        onnx_files.append(os.path.join(root, file))
            
            if onnx_files:
                print(f"✅ {item}")
                for onnx_file in onnx_files:
                    # get relative path from models directory
                    rel_path = os.path.relpath(onnx_file, models_dir)
                    print(f"   📄 {rel_path}")
            else:
                print(f"⚠️  {item} (no ONNX file found)")
    
    print("=" * 50)


def visualize_detections(
    image_path: str, 
    detections: List[Tuple], 
    threshold: float = 0.3,
    save_path: str = None
    ):
    """
    Visualize detections on image using configuration.
    
    Args:
        image_path: Path to image file or S3 path
        detections: List of detections from predict_image()
        threshold: Confidence threshold for display
        save_path: S3 path to save visualization (e.g., s3://bucket/key.png)
    """
    # load imag from s3 or local path
    if image_path.startswith('s3://'):
        bucket_key = image_path.replace('s3://', '').split('/', 1)
        bucket_name = bucket_key[0]
        s3_key = bucket_key[1]
        
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(response['Body'].read())
            img = mpimg.imread(tmp_file.name)
        os.unlink(tmp_file.name)
    else:
        img = mpimg.imread(image_path)
    
    plt.figure(figsize=[12, 8])
    plt.imshow(img)
    
    width = img.shape[1]
    height = img.shape[0]
    colors = {}
    num_detections = 0
    
    for det in detections:
        if len(det) < 6:
            continue
            
        klass, score, x0, y0, x1, y1 = det
        
        if score < threshold:
            continue
            
        num_detections += 1
        cls_id = int(klass)
        
        # generate random color for each class
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        
        # convert normalized coordinates to pixel coordinates
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        
        # draw bounding box
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5
        )
        plt.gca().add_patch(rect)
        
        # add confidence score
        plt.gca().text(
            xmin,
            ymin - 2,
            f"{score:.3f}",
            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
            fontsize=12,
            color="white"
        )
    
    plt.title(f"Plastic Bag Detection - {num_detections} detections found")
    plt.axis('off')
    
    if save_path:
        if save_path.startswith('s3://'):
            # save to s3
            bucket_key = save_path.replace('s3://', '').split('/', 1)
            bucket_name = bucket_key[0]
            s3_key = bucket_key[1]
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                plt.savefig(tmp_file.name, bbox_inches='tight', dpi=150)
                tmp_file_path = tmp_file.name
            
            try:
                s3_client = boto3.client('s3')
                s3_client.upload_file(tmp_file_path, bucket_name, s3_key)
                print(f"Visualization saved to: {save_path}")
                os.unlink(tmp_file_path)
            except Exception as e:
                print(f"Error uploading to S3: {e}")
                os.unlink(tmp_file_path)
        else:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
    
    plt.close()
    print(f"Number of detections: {num_detections}")



def get_s3_images(bucket: str, prefix: str, max_images: int = 10) -> list[dict]:
    """
    Get list of image files from S3 bucket and return list of dicts with image info.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix
        max_images: maximum number of images to return
    Returns:
        list of dicts with image info

        Example:
        [
            {
                'key': 'path/to/image.jpg',
                'size': 1024,
                'last_modified': '2021-01-01T00:00:00Z'
            },
            {
                'key': 'path/to/image2.jpg',
                'size': 2048,
                'last_modified': '2021-01-01T00:00:00Z'
            },
            ...
        ]
    """
    s3_client = boto3.client('s3')
    
    try:
        print(f"Listing images in s3://{bucket}/{prefix}...")
        
        # List objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if it's an image file
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        images.append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified']
                        })
        
        # Sort by name and limit
        images.sort(key=lambda x: x['key'])
        images = images[:max_images]
        
        print(f"Found {len(images)} image files:")
        for i, img in enumerate(images):
            filename = img['key'].split('/')[-1]  # Get just the filename
            size_kb = img['size'] / 1024
            print(f"  {i+1:2d}. {filename} ({size_kb:.1f} KB)")
        
        return images
        
    except Exception as e:
        print(f"Error listing S3 images: {e}")
        return []


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        raise Exception(f"Error reading image: {e}")