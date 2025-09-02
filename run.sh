#!/bin/bash

set -e

# Deploy YOLO Lambda function
deploy_lambda() {
    echo "Starting YOLO Lambda deployment..."
    
    # Copy config.yaml from root to Lambda app directory
    echo "Copying config.yaml..."
    cp config.yaml src/sagemaker/lambda/yolo-inference/app/config.yaml
    
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

# Call the appropriate function based on argument
case "${1:-deploy}" in
    deploy)
        deploy_lambda
        ;;
    inference)
        run_inference
        ;;
esac

