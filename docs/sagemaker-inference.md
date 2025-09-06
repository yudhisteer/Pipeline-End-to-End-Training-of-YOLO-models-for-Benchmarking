# Deployment and Inference

## Plan of Action
0. [Prerequisites](#0-prerequisites)
1. [Local Inference with local model](#1-local-inference-with-local-model)

------------------------------------------------------

<a name="0-prerequisites"></a>
## 0. Prerequisites
To serve a model on SageMaker, our trained model (.onnx or .pt) needs to be packaged in a format that SageMaker can deploy. We create a `tar.gz` file that contains the model weights and an `inference.py` file that will be used to perform inference.
The inference.py should contain functions that are used by SageMaker to perform inference:

- `model_fn`: Load the model and return it.
- `input_fn`: Preprocess the input data.
- `predict_fn`: Perform inference.
- `output_fn`: Postprocess the output data.
- `transform_fn`: Alternative to the above three functions - handles the entire request/response cycle in a single function.



 ### 0.1. Load a Model with model_fn()
After we trained our model, we save it to `.pt` file which means we serialize it (convert to bytes/file).

 ```python
# During training:
torch.save(model, 'model.pt')  # Serialization/saving
```

In order to do inference, we need to load  the model, meaning read the saved model files from disk and reconstruct the model object in memory so it can be used for making predictions.

 ```python
def model_fn(model_dir) -> torch.nn.Module:
    model_path = os.path.join(model_dir, 'best.pt')
    
    # This line does the "loading":
    # 1. Reads the .pt file from disk
    # 2. Deserializes the binary data
    # 3. Reconstructs the neural network in memory
    # 4. Restores all the trained weights/parameters
    # 5. Creating a usable Python object that can make predictions
    model = torch.load(model_path)
    
    # Now 'model' is a live Python object that can:
    # - Accept input data
    # - Run forward passes
    # - Make predictions
    
    return model
 ```

In summary, before loading the model it exists only as files on disk (inactive). After loading the model it exists as a Python object in RAM (active and ready for inference).



### 0.2. Process Model Input with input_fn(request_body, request_content_type)

The `input_fn` function will take request data and deserializes the data into an object for prediction.

When an `InvokeEndpoint` operation is made against an Endpoint running a SageMaker PyTorch model server, the model server receives two pieces of information:

- The request Content-Type, for example “application/x-npy”
- The request data body, a byte array

`input_fn()` acts as the data preparation layer between the incoming request and your model:

- Takes raw data from HTTP requests
- Converts it into the proper format/structure your model expects!
- Handles different input content types (JSON, CSV, image bytes, etc.)
- Performs any necessary preprocessing (normalization, resizing, tokenization, etc.)

### Example of JSON Input for Image Classification

```python
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io

def input_fn(request_body, request_content_type) -> torch.Tensor:
    """
    Preprocess input data before feeding it to the model
    
    Args:
        request_body: The raw input data from the request
        request_content_type: MIME type of the input data (e.g., 'application/json')
        
    Returns:
        processed_input: Data in the format expected by your model
    """
    if request_content_type == 'application/json':
        # Parse JSON request
        input_data = json.loads(request_body)
        
        # Decode base64 image
        image_data = base64.b64decode(input_data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Apply preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to tensor and add batch dimension
        processed_image = transform(image).unsqueeze(0)
        return processed_image
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
```

SageMaker calls this function for every inference request, before calling `predict_fn()`. Note that `input_fn()` is optional - if we don't provide it, SageMaker will use default input handling based on your framework. This function deserializes `JSON, CSV, or NPY` encoded data into a `torch.Tensor`. However, custom `input_fn()` gives us full control over data preprocessing.

### 0.3. Get Predictions with predict_fn(input_object, model)

After the **inference request** has been **deserialized** by `input_fn`, the SageMaker PyTorch model server invokes `predict_fn` on the return value of `input_fn`.

`predict_fn()` is the core function that takes the preprocessed input data from `input_fn()` and runs inference using our loaded model from `model_fn()` to generate predictions.

Below is an example of `predict_fn()` for a classification model.

```python
import torch

def predict_fn(input_data, model) -> dict:
    """
    Run inference on the preprocessed input data
    
    Args:
        input_data: Preprocessed data from input_fn(). Default is torch.Tensor
        model: Loaded model from model_fn()
        
    Returns:
        predictions: Raw output from the model (e.g., probabilities or class labels)
        Default is torch.Tesnor
    """

    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Run forward pass
        outputs = model(input_data)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get predicted classes
        predicted_classes = torch.argmax(probabilities, dim=1)
    
    return {
        'probabilities': probabilities.cpu().numpy(),
        'predictions': predicted_classes.cpu().numpy()
    }
```

Like `input_fn()`, `predict_fn()` is optional. If not provided, SageMaker uses framework-specific default prediction logic. 

```mermaid
input_fn() → predict_fn() → output_fn()
    ↓            ↓             ↓
Preprocessed → Raw Model → Formatted
   Data        Output      Response
```


### 0.4. Process Model Output with output_fn(prediction, content_type)

After `predict_fn()` generates predictions, the SageMaker PyTorch model server invokes `output_fn()` on the return value of `predict_fn()`.

`output_fn()` is responsible for post-processing the raw model predictions and formatting them into a proper `HTTP response` that can be sent back to the client.

Below is an example of `output_fn()` for a classification model.

```python
import json
import numpy as np

def output_fn(prediction, content_type) -> str:
    """
    Post-process model predictions and format the response
    
    Args:
        prediction: Raw output from predict_fn()
        content_type: MIME type for the response (e.g., 'application/json')
        
    Returns:
        response: Formatted response data
    """
    if content_type == 'application/json':
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(prediction['probabilities'], np.ndarray):
            prediction['probabilities'] = prediction['probabilities'].tolist()
        if isinstance(prediction['predictions'], np.ndarray):
            prediction['predictions'] = prediction['predictions'].tolist()
        
        # Format the response
        response = {
            'statusCode': 200,
            'predictions': prediction['predictions'],
            'confidence_scores': prediction['probabilities'],
            'timestamp': str(datetime.now())
        }
        
        return json.dumps(response)
    
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
```

The default implementation expects prediction to be a torch.Tensor and can serialize the result to `JSON, CSV, or NPY`. It accepts response content types of `application/json`, `text/csv`, and `application/x-npy`.


```mermaid
predict_fn() → output_fn() → HTTP Response
     ↓            ↓              ↓
Raw Model   → Formatted    → Client
  Output       Response       Receives
```


### 0.5. transform_fn()

`transform_fn()` is an alternative function that handles the entire inference pipeline in a single function, replacing the combination of `input_fn()` + `predict_fn()` + `output_fn()`.

```mermaid
# Separate functions approach:
Client Request → input_fn() → predict_fn() → output_fn() → Response

# Transform function approach:
Client Request → transform_fn() → Response
```


### Example  for Image Classification:

```python
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
import io

def transform_fn(model, request_body, request_content_type, accept):
    # Input processing
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Decode and preprocess image
        image_data = base64.b64decode(input_data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        processed_image = transform(image).unsqueeze(0)
        
        # Model inference
        model.eval()
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Output formatting
        if accept == 'application/json':
            response = {
                'predicted_class': int(predicted_class[0]),
                'confidence': float(probabilities[0][predicted_class[0]]),
                'all_probabilities': probabilities[0].tolist()
            }
            return json.dumps(response)
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
```

We mostly choose to work with sepqarate functions as it is more modular and suitable for complex pipelines.



-----------------------------------------
<a name="1-local-inference-with-local-model"></a>
## 1. Local Inference with local model











-----------------------------------------

## References
1. https://docs.aws.amazon.com/sagemaker/latest/dg/neo-deployment-hosting-services-prerequisites.html
2. https://sagemaker.readthedocs.io/en/v1.40.0/using_pytorch.html