"""
Script to test YOLO Lambda inference
"""
import json
import boto3
from datetime import datetime
from typing import Dict, List, Any
import requests

    
from utils.utils_config import load_config, get_lambda_config
from utils.utils_inference import get_s3_images


def test_health_check(base_url: str, health_endpoint: str = '/health') -> bool:
    """Test the health check endpoint"""
    print(f"Testing health check at {base_url}{health_endpoint}...")
    try:
        response = requests.get(f"{base_url}{health_endpoint}", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False


def run_batch_inference(
    s3_bucket: str, 
    s3_prefix: str, 
    base_url: str, 
    confidence_threshold: float, 
    predict_endpoint: str, 
    num_images: int = 5,
    save_to_s3: bool = True
    ) -> bool:

    """Test YOLO batch inference with S3 images"""
    print(f"\nTesting YOLO batch inference with up to {num_images} real images from S3...")
    
    # get image files
    image_files = get_s3_images(s3_bucket, s3_prefix, num_images)
    
    if not image_files:
        print("No image files found in S3 bucket")
        return False
    
    # create batch payload
    images = []
    for img_info in image_files:
        filename = img_info['key'].split('/')[-1]
        images.append({
            "image_id": filename,
            "s3_bucket": s3_bucket,
            "s3_key": img_info['key']
        })
    
    payload = {
        "images": images,
        "confidence_threshold": confidence_threshold,
        "chunk_size": 3
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending batch request for {len(images)} images...")
    
    try:
        response = requests.post(
            f"{base_url}{predict_endpoint}",
            json=payload,
            headers=headers,
            timeout=300
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if save_to_s3:
                s3_key = save_results_to_s3(result, s3_bucket)
                print(f"\nResults saved to: s3://{s3_bucket}/{s3_key}")
            else:
                print("Batch inference successful!")
                print(f"Batch size: {result.get('batch_size', 0)}")
                print(f"Chunks processed: {result.get('chunks_processed', 0)}")
                print(f"Total detections: {result.get('total_detections', 0)}")
                
                for img_result in result.get('results', []):
                    print(f"\n{img_result['image_id']}:")
                    print(f"  Status: {img_result['status']}")
                    if img_result['status'] == 'success':
                        num_det = img_result['num_detections']
                        print(f"  Detections: {num_det}")
                        
                        if num_det > 0 and 'detections' in img_result:
                            for i, det in enumerate(img_result['detections']):
                                conf = det.get('final_confidence', 0)
                                bbox = det.get('bbox_corners', [])
                                print(f"    Det {i+1}: conf={conf:.3f}, bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")
                    else:
                        print(f"  Error: {img_result.get('error', 'Unknown error')}")
            
            return True
        else:
            print("Batch inference failed!")
            try:
                error_response = response.json()
                print(f"Error: {error_response}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except Exception as e:
        print(f"Batch request failed: {e}")
        return False








def save_inference_results_to_s3(
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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


def save_results_to_s3(
    batch_result: Dict[str, Any], 
    s3_bucket: str,
    ) -> str:
    """
    Print results in the console AND save to S3
    
    Args:
        batch_result: The full batch inference result from your API
        s3_bucket: S3 bucket to save results
    
    Returns:
        S3 key where results were saved
    """
    results = batch_result.get('results', [])
    
    # Print results in your preferred format
    print(f"\nBatch inference completed!")
    print(f"Batch size: {batch_result.get('batch_size', 0)}")
    print(f"Chunks processed: {batch_result.get('chunks_processed', 0)}")
    print(f"Total detections: {batch_result.get('total_detections', 0)}")
    
    for img_result in results:
        print(f"\n{img_result['image_id']}:")
        print(f"  Status: {img_result['status']}")
        
        if img_result['status'] == 'success':
            num_det = img_result['num_detections']
            print(f"  Detections: {num_det}")
            
            # Show all detections
            if num_det > 0 and 'detections' in img_result:
                for i, det in enumerate(img_result['detections']):
                    conf = det.get('final_confidence', 0)
                    bbox = det.get('bbox_corners', [])
                    print(f"    Det {i+1}: conf={conf:.3f}, bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")
        else:
            print(f"  Error: {img_result.get('error', 'Unknown error')}")
    
    s3_key = save_inference_results_to_s3(results, s3_bucket)
    
    return s3_key




def main():
    
    # load configs
    config = load_config("config.yaml")
    lambda_config = get_lambda_config(config)
    
    # test api local or prod
    base_url = lambda_config['base_url'] # prod url

    confidence_threshold = lambda_config['confidence_threshold']
    
    # 1. Test health check first
    if not test_health_check(base_url, '/health'):
        return
    print("Health check passed!")
    
    # 2. run batch inference
    test_config = config.get('inference', {}).get('test', {})
    success_batch = run_batch_inference(
        test_config.get('s3_bucket'),
        test_config.get('s3_prefix'), 
        base_url,
        confidence_threshold,
        lambda_config.get('predict_endpoint', '/predict'),
        test_config.get('batch_size', 5),
        save_to_s3=True,
    )
    print(f"\nResults: Batch={success_batch}")

if __name__ == "__main__":
    main()