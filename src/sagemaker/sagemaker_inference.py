"""
Script to test YOLO Lambda inference
"""
import requests

    
from utils.utils_config import load_config, get_lambda_config
from utils.utils_inference import get_s3_images


def test_health_check(base_url: str, health_endpoint: str) -> bool:
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
    num_images: int = 5
    ) -> bool:

    """Test YOLO batch inference with S3 images"""
    print(f"\nTesting YOLO batch inference with up to {num_images} real images from S3...")
    
    # Get image files
    image_files = get_s3_images(s3_bucket, s3_prefix, num_images)
    
    if not image_files:
        print("No image files found in S3 bucket")
        return False
    
    # Create batch payload
    images = []
    for img_info in image_files:
        filename = img_info['key'].split('/')[-1]  # Get just the filename for ID
        images.append({
            "image_id": filename,
            "s3_bucket": s3_bucket,
            "s3_key": img_info['key']
        })
    
    payload = {
        "images": images,
        "confidence_threshold": confidence_threshold,
        "chunk_size": 3  # Process 3 images per chunk
    }
    
    headers = {"Content-Type": "application/json"}
    
    print(f"Sending batch request for {len(images)} images...")
    
    try:
        response = requests.post(
            f"{base_url}{predict_endpoint}",
            json=payload,
            headers=headers,
            timeout=300  # 5 minutes for batch processing
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Batch inference successful!")
            print(f"Batch size: {result.get('batch_size', 0)}")
            print(f"Chunks processed: {result.get('chunks_processed', 0)}")
            print(f"Total detections: {result.get('total_detections', 0)}")
            
            # Print results for each image
            for img_result in result.get('results', []):
                print(f"\n{img_result['image_id']}:")
                print(f"  Status: {img_result['status']}")
                if img_result['status'] == 'success':
                    num_det = img_result['num_detections']
                    print(f"  Detections: {num_det}")
                    
                    # Show detections if available
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


def main():
    print("YOLO Lambda Inference Test")
    print("=" * 40)
    
    # Load configuration
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
        test_config.get('batch_size', 5)
    )
    
    print(f"\nResults: Batch={success_batch}")

if __name__ == "__main__":
    main()