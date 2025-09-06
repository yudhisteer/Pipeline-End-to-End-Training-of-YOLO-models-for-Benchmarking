.PHONY: deploy inference train finetuning pipeline

# Deploy YOLO Lambda function
deploy:
	@./run.sh deploy

# Run YOLO inference
inference:
	@./run.sh inference

# Run YOLO training
train:
	@./run.sh train

# Run YOLO finetuning
finetuning:
	@./run.sh finetuning

# Run YOLO pipeline
pipeline:
	@./run.sh pipeline