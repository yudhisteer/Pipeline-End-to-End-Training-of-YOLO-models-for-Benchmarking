#!/bin/bash

set -e

# Deploy YOLO Lambda function
deploy_lambda() {
    echo "Starting YOLO Lambda deployment..."
    
    # Copy config.yaml from root to Lambda app directory
    echo "Copying config.yaml..."
    cp config.yaml src/sagemaker/lambda/yolo-inference/app/config.yaml
    
    # Extract and print job_name from Lambda app config.yaml (deployment section)
    job_name=$(grep -A 5 "deployment:" src/sagemaker/lambda/yolo-inference/app/config.yaml | grep "job_name:" | sed 's/.*job_name: *"\([^"]*\)".*/\1/')
    echo "Job name: $job_name"
    
    # Change to Lambda directory
    cd src/sagemaker/lambda/yolo-inference
    
    # Build and deploy
    echo "Building Lambda function..."
    sam build
    
    echo "Deploying Lambda function..."
    sam deploy --no-confirm-changeset #confirm without user confirmation
    
    # Return to project root
    cd ../../../..
    
    echo "YOLO Lambda function deployed successfully!"
}

# Run YOLO inference
run_inference() {
    echo "Running YOLO inference..."
    python src/sagemaker/sagemaker_inference.py
    echo "Inference completed!"
}

# Run YOLO training
run_training() {
    echo "Running YOLO training..."
    python src/sagemaker/sagemaker_pipeline.py
    echo "Training completed!"
}

# Call the appropriate function based on argument
case "${1:-deploy}" in
    deploy)
        deploy_lambda
        ;;
    inference)
        run_inference
        ;;
    train)
        run_training
        ;;
esac

