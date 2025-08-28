"""
Script to test YOLO Lambda inference
"""

import requests
    
from utils.utils_config import load_config, get_inference_config
from utils.utils_inference import get_s3_images, save_results_and_visualizations


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
    s3_inference_dataset: str, 
    base_url: str, 
    confidence_threshold: float, 
    predict_endpoint: str, 
    num_images: int = 5,
    save_to_s3: bool = True,
) -> bool:
    """YOLO batch inference"""
    print(f"\nTesting YOLO batch inference with up to {num_images} real images from S3...")
    
    # extract bucket and prefix from S3 path
    s3_path = s3_inference_dataset[5:]  # remove 's3://'
    parts = s3_path.split('/', 1)
    
    if len(parts) < 2:
        print(f"Invalid S3 path format: {s3_inference_dataset}")
        return False
    
    s3_bucket = parts[0]
    s3_prefix = parts[1]
    
    print(f"Using S3 bucket: {s3_bucket}")
    print(f"Using S3 prefix: {s3_prefix}")
    
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
            
            # save results and visualizations
            if save_to_s3:
                saved_files = save_results_and_visualizations(
                    result,
                    s3_bucket,
                    s3_prefix,
                    confidence_threshold
                )
                
                print(f"\nFiles saved:")
                print(f"  JSON: s3://{s3_bucket}/{saved_files['json_key']}")
                
                if saved_files['visualization_keys']:
                    print(f"  Visualizations: {len(saved_files['visualization_keys'])} images")
            
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




def main():
    
    # load configs
    config = load_config("config.yaml")
    inference_config = get_inference_config(config)
    lambda_config = inference_config.get('lambda', {})
    base_url = lambda_config['base_url'] # prod url
    confidence_threshold = inference_config['confidence_threshold']
    batch_size = inference_config['batch_size']
    s3_inference_dataset = inference_config['s3_inference_dataset']
    predict_endpoint = lambda_config['endpoints']['predict']
    
    # 1. Test health check first
    if not test_health_check(base_url, '/health'):
        return
    print("Health check passed!")
    
    # 2. run batch inference
    success = run_batch_inference(
        s3_inference_dataset=s3_inference_dataset,
        base_url=base_url,
        confidence_threshold=confidence_threshold,
        predict_endpoint=predict_endpoint,
        num_images=batch_size,
        save_to_s3=True,
    )
    print(f"\nResults: Batch={success}")

if __name__ == "__main__":
    main()