import matplotlib.pyplot as plt
import os
import random
import tempfile
import boto3
import base64       
from typing import List, Tuple, Dict, Any, Union, Optional  
from datetime import datetime
import json
import matplotlib.image as mpimg
import numpy as np

# we use the same timestamp
GLOBAL_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def list_local_models():
    """List all locally available models organized by training job."""
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print("No local models found.")
        return
    
    print("Locally Available Models:")
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
                print(f"{item}")
                for onnx_file in onnx_files:
                    # get relative path from models directory
                    rel_path = os.path.relpath(onnx_file, models_dir)
                    print(f"   📄 {rel_path}")
            else:
                print(f" {item} (no ONNX file found)")
    
    print("=" * 50)





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
        
        # list objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # check if it's an image file
                    if any(key.lower().endswith(ext) for ext in image_extensions):
                        images.append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified']
                        })
        
        # sort by name and limit
        images.sort(key=lambda x: x['key'])
        images = images[:max_images]
        
        print(f"Found {len(images)} image files:")
        for i, img in enumerate(images):
            filename = img['key'].split('/')[-1]  # get just the filename
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





def create_visualization(
    image_path: str,
    detections: Union[List[Tuple], List[Dict]],
    threshold: float = 0.3,
    save_path: Optional[str] = None,
    title: str = "Object Detection"
) -> bool:
    """
    Unified function to visualize detections on image with support for multiple formats.
    
    Args:
        image_path: Path to image file (local or S3 path like s3://bucket/key)
        detections: List of detections either as:
            - List[Dict]: With keys 'class_id', 'final_confidence', 'bbox_corners' (pixel coords)
            - List[Tuple]: Format (class_id, confidence, x0_norm, y0_norm, x1_norm, y1_norm)
        threshold: Confidence threshold for display (default: 0.3)
        save_path: Optional path to save visualization (local or S3)
        title: Title for the visualization (default: "Object Detection")
    
    Returns:
        bool: True if successful, False if error occurred
    """
    fig = None
    temp_files = []
    
    try:
        # Load image from S3 or local path
        img = load_image_from_path(image_path, temp_files)
        
        # Get image dimensions
        height, width = img.shape[0], img.shape[1]
        
        # Convert detections to unified format if needed
        normalized_detections = normalize_detection_format(detections, width, height)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=[12, 8])
        ax.imshow(img)
        
        # Draw detections
        num_detections = draw_detections(ax, normalized_detections, width, height, threshold)
        
        # Set title and formatting
        ax.set_title(f"{title} - {num_detections} detections found")
        ax.axis('off')
        
        # Save if path provided
        if save_path:
            save_success = save_figure(fig, save_path, temp_files)
            if save_success:
                print(f"Visualization saved to: {save_path}")
            else:
                print(f"Failed to save visualization to: {save_path}")
        
        print(f"Number of detections: {num_detections}")
        return True
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        return False
        
    finally:
        # Clean up
        if fig is not None:
            plt.close(fig)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def load_image_from_path(image_path: str, temp_files: List[str]) -> np.ndarray:
    """
    Load image from S3 or local path.
    
    Args:
        image_path: Path to image (local or S3)
        temp_files: List to track temporary files for cleanup
    
    Returns:
        numpy array of the image
    """
    if image_path.startswith('s3://'):
        # Parse S3 path
        path_parts = image_path.replace('s3://', '').split('/', 1)
        bucket_name = path_parts[0]
        s3_key = path_parts[1] if len(path_parts) > 1 else ''
        
        # Get file extension from the S3 key
        file_ext = os.path.splitext(s3_key)[1] if s3_key else '.jpg'
        if not file_ext:
            file_ext = '.jpg'  # Default to .jpg if no extension
        
        # Download from S3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        
        # Save to temporary file with correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(response['Body'].read())
            temp_files.append(tmp_file.name)
            return mpimg.imread(tmp_file.name)
    else:
        # Load from local path
        return mpimg.imread(image_path)


def normalize_detection_format(
    detections: Union[List[Tuple], List[Dict]], 
    image_width: int, 
    image_height: int
    ) -> List[Tuple]:
    """
    Convert detections to normalized tuple format.
    
    Args:
        detections: Input detections in either format
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        List of tuples: (class_id, confidence, x0_norm, y0_norm, x1_norm, y1_norm)
    """
    if not detections:
        return []
    
    # Check if already in tuple format
    if isinstance(detections[0], tuple):
        # Validate tuple format
        validated = []
        for det in detections:
            if len(det) >= 6:
                validated.append(det)
        return validated
    
    # Convert from dict format
    converted = []
    for det in detections:
        class_id = det.get('class_id', 0)
        confidence = det.get('final_confidence', 0)
        bbox_corners = det.get('bbox_corners', [0, 0, 0, 0])
        
        if len(bbox_corners) >= 4:
            x1, y1, x2, y2 = bbox_corners[:4]
            
            # Convert pixel coordinates to normalized coordinates
            x0_norm = x1 / image_width
            y0_norm = y1 / image_height
            x1_norm = x2 / image_width
            y1_norm = y2 / image_height
            
            # Clamp to [0, 1] range
            x0_norm = max(0, min(1, x0_norm))
            y0_norm = max(0, min(1, y0_norm))
            x1_norm = max(0, min(1, x1_norm))
            y1_norm = max(0, min(1, y1_norm))
            
            converted.append((class_id, confidence, x0_norm, y0_norm, x1_norm, y1_norm))
    
    return converted


def draw_detections(
    ax: plt.Axes,
    detections: List[Tuple],
    width: int,
    height: int,
    threshold: float
    ) -> int:
    """
    Draw bounding boxes and confidence scores on the axes.
    
    Args:
        ax: Matplotlib axes object
        detections: Normalized detections
        width: Image width in pixels
        height: Image height in pixels
        threshold: Confidence threshold
    
    Returns:
        Number of detections drawn
    """
    colors = {}
    num_detections = 0
    
    for det in detections:
        if len(det) < 6:
            continue
        
        class_id, score, x0, y0, x1, y1 = det
        
        # Ensure score is a valid number for comparison
        try:
            score = float(score)
        except (ValueError, TypeError):
            continue  # Skip invalid detections
        
        if score < threshold:
            continue
        
        num_detections += 1
        cls_id = int(class_id)
        
        # Generate consistent random color for each class
        if cls_id not in colors:
            # Use a seed for consistent colors across runs
            random.seed(cls_id)
            colors[cls_id] = (random.random(), random.random(), random.random())
        
        # Convert normalized coordinates to pixel coordinates
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        
        # Draw bounding box
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=colors[cls_id],
            linewidth=3.5
        )
        ax.add_patch(rect)
        
        # Add confidence score label
        ax.text(
            xmin,
            ymin - 2,
            f"{score:.3f}",
            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
            fontsize=12,
            color="white"
        )
    
    return num_detections


def save_figure(fig: plt.Figure, save_path: str, temp_files: List[str]) -> bool:
    """
    Save figure to local path or S3.
    
    Args:
        fig: Matplotlib figure object
        save_path: Path to save (local or S3)
        temp_files: List to track temporary files
    
    Returns:
        bool: Success status
    """
    try:
        if save_path.startswith('s3://'):
            # Save to S3
            path_parts = save_path.replace('s3://', '').split('/', 1)
            bucket_name = path_parts[0]
            s3_key = path_parts[1] if len(path_parts) > 1 else ''
            
            # Save to temporary file first
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                fig.savefig(tmp_file.name, bbox_inches='tight', dpi=150)
                temp_files.append(tmp_file.name)
                
                # Upload to S3
                s3_client = boto3.client('s3')
                s3_client.upload_file(tmp_file.name, bucket_name, s3_key)
            
            return True
        else:
            # Save to local path
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            return True
            
    except Exception as e:
        print(f"Error saving figure: {e}")
        return False



def save_batch_inference_results_to_s3(
    results: List[Dict[str, Any]], 
    s3_bucket: str,
    base_folder: str = "yolo-pipeline/Inference", #TODO: take from config
    ) -> str:
    """
    Save inference results to S3 in JSON format
    
    Args:
        results: List of inference results from the batch processing
        s3_bucket: S3 bucket name
        base_folder: Base folder name in S3 (default: "Inference")
    
    Returns:
        S3 key where the file was saved
    """
    s3_client = boto3.client('s3')
    
    # generate timestamp for folder name
    timestamp = GLOBAL_TIMESTAMP
    
    # create S3 key
    s3_key = f"{base_folder}/{timestamp}/output.json"
    
    # transform results
    formatted_results = {}
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_images": len(results),
        "successful_images": 0,
        "failed_images": 0,
        "total_detections": 0
    }
    
    for result in results:
        image_id = result.get('image_id', 'unknown')
        status = result.get('status', 'unknown')
        
        if status == 'success':
            summary["successful_images"] += 1
            num_detections = result.get('num_detections', 0)
            summary["total_detections"] += num_detections
            
            # format detections
            detections_list = []
            if 'detections' in result:
                for i, detection in enumerate(result['detections']):
                    conf = detection.get('final_confidence', 0)
                    bbox = detection.get('bbox_corners', [0, 0, 0, 0])
                    
                    detections_list.append({
                        "detection_id": i + 1,
                        "confidence": round(conf, 3),
                        "bbox_corners": [int(coord) for coord in bbox],
                        "bbox_center": detection.get('bbox_center', []),
                        "objectness": detection.get('objectness', 0),
                        "class_confidence": detection.get('class_confidence', 0),
                        "class_id": detection.get('class_id', 0),
                        "class_scores": detection.get('class_scores', [])
                    })
            
            formatted_results[image_id] = {
                "status": status,
                "num_detections": num_detections,
                "detections": detections_list
            }
        else:
            summary["failed_images"] += 1
            formatted_results[image_id] = {
                "status": status,
                "error": result.get('error', 'Unknown error'),
                "num_detections": 0,
                "detections": []
            }
    
    # create final JSON structure
    output_data = {
        "summary": summary,
        "results": formatted_results
    }
    
    try:
        # convert to JSON string
        json_content = json.dumps(output_data, indent=2, ensure_ascii=False)
        
        # upload to S3
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=json_content,
            ContentType='application/json'
        )
        
        print(f"Results saved to S3: s3://{s3_bucket}/{s3_key}")
        print(f"Summary: {summary['successful_images']}/{summary['total_images']} images processed successfully")
        print(f"Total detections: {summary['total_detections']}")
        
        return s3_key
        
    except Exception as e:
        print(f"Error saving results to S3: {e}")
        raise e





def save_batch_inference_visualizations(
    batch_results: List[Dict[str, Any]], 
    s3_bucket: str,
    s3_images_prefix: str,
    confidence_threshold: float = 0.3
    ) -> List[str]:
    """Save all visualizations - simple, one by one"""
    
    timestamp = GLOBAL_TIMESTAMP
    visualization_keys = []
    
    print(f"\nCreating visualizations...")
    print(f"Saving to: s3://{s3_bucket}/yolo-pipeline/Inference/{timestamp}/Images/")
    
    for i, result in enumerate(batch_results):
        image_id = result.get('image_id', 'unknown')
        
        # Skip failed or empty results
        if result.get('status') != 'success':
            print(f"  [{i+1}] Skipped {image_id}: {result.get('status')}")
            continue
            
        detections = result.get('detections', [])
        if not detections:
            print(f"  [{i+1}] Skipped {image_id}: no detections")
            continue
        
        # Create paths
        original_image_s3_path = f"s3://{s3_bucket}/{s3_images_prefix}/{image_id}"
        vis_s3_key = f"yolo-pipeline/Inference/{timestamp}/Images/{image_id}"
        vis_s3_path = f"s3://{s3_bucket}/{vis_s3_key}"
        
        # Create visualization
        success = create_visualization(
            original_image_s3_path,
            detections,
            confidence_threshold,
            vis_s3_path
        )
        
        if success:
            visualization_keys.append(vis_s3_key)
            print(f"  [{i+1}] {image_id}: saved")
        else:
            print(f"  [{i+1}] {image_id}: failed")
    
    print(f"\n Created {len(visualization_keys)} visualizations")
    print("*"*50)
    return visualization_keys


def save_results_and_visualizations(
    batch_result: Dict[str, Any],
    s3_bucket: str,
    s3_images_prefix: str,
    confidence_threshold: float = 0.3
    ) -> Dict[str, Any]:
    """Save JSON results and create visualizations"""
    
    results = batch_result.get('results', [])
    json_s3_key = save_batch_inference_results_to_s3(results, s3_bucket)
    
    # 2. Create visualizations
    visualization_keys = save_batch_inference_visualizations(
        results, 
        s3_bucket, 
        s3_images_prefix, 
        confidence_threshold
    )
    
    return {
        'json_key': json_s3_key,
        'visualization_keys': visualization_keys
    }
