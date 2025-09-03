# YOLO Pipeline
.PHONY: deploy inference train

# Deploy YOLO Lambda function
deploy:
	@./run.sh deploy

# Run YOLO inference
inference:
	@./run.sh inference

# Run YOLO training
training:
	@./run.sh train