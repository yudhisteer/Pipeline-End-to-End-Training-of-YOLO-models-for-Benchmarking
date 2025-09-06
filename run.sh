#!/bin/bash

set -e

# Deploy YOLO Lambda function
deploy_lambda() {
    echo "Starting YOLO Lambda deployment..."
    
    # copy config.yaml from root to Lambda app directory
    echo "Copying config.yaml..."
    cp config.yaml src/sagemaker/lambda/yolo-inference/app/config.yaml
    
    # extract and print job_name from Lambda app config.yaml
    job_name=$(grep -A 5 "deployment:" src/sagemaker/lambda/yolo-inference/app/config.yaml | grep "job_name:" | sed "s/.*job_name: *['\"]\\([^'\"]*\\)['\"].*/\\1/")
    if [ -z "$job_name" ]; then
        echo "Job name empty, getting latest successful job name from pipeline"
    else
        echo "Job name: $job_name"
    fi
    
    # change to Lambda directory
    cd src/sagemaker/lambda/yolo-inference
    
    # build and deploy
    echo "Building Lambda function..."
    sam build
    
    echo "Deploying Lambda function..."
    sam deploy --no-confirm-changeset #confirm without user confirmation
    
    # return to project root
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
    python src/sagemaker/sagemaker_train.py
    echo "Training completed!"
}

# Run YOLO finetuning
run_finetuning() {
    echo "Running YOLO finetuning..."
    python src/sagemaker/sagemaker_hyperparameter_tuning.py
    echo "Finetuning completed!"
}

# Run YOLO pipeline
run_pipeline() {
    echo "Running YOLO pipeline..."

    echo "1. Convert dataset to COCO format..."
    python src/sagemaker/sagemaker_data_manager.py

    echo "2. Running YOLO finetuning..."
    python src/sagemaker/sagemaker_hyperparameter_tuning.py

    echo "3. Running YOLO training..."
    python src/sagemaker/sagemaker_train.py

    echo "4. Running YOLO evaluation..."
    python src/sagemaker/sagemaker_evaluate.py

    echo "5. Running YOLO inference..."
    python src/sagemaker/sagemaker_inference.py

    echo "Pipeline completed!"
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
    finetuning)
        run_finetuning
        ;;
    pipeline)
        run_pipeline
        ;;
esac

