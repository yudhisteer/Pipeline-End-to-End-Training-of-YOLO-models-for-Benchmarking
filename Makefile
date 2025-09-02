# YOLO Pipeline
.PHONY: deploy inference

# Deploy YOLO Lambda function
deploy:
	@./run.sh deploy

# Run YOLO inference
inference:
	@python src/sagemaker/sagemaker_inference.py