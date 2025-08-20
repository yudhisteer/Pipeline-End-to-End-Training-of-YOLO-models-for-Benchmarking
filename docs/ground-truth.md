# Ground Truth Labeling on SageMaker AI

This documentation provide instructions on how to create a a Ground Truth labeling job on SageMaker AI. We provide some helper functions that will process the annotations into YOLO format for model training. The code and instructions associated to this README is based from the official Amazon documentation with references included at the bottom.

## Plan of Action
1. [Setting up our S3 bucket](#1-setting-up-our-s3-bucket)
2. [Creating the manifest file](#2-creating-the-manifest-file)
3. [Create SageMaker Ground Truth job](#3-create-sagemaker-ground-truth-job)
4. [Start Labeling](#4-start-labeling)
5. [Parse the annotations](#5-parse-the-annotations)
6. [Transform the annotations into YOLO format](#6-transform-the-annotations-into-yolo-format)
7. [Visualize YOLO Annotations](#7-visualize-yolo-annotations)


----------------------------------
<a name="1-setting-up-our-s3-bucket"></a>
## 1. Setting up our S3 bucket

We start by creating an S3 bucket - `ground-truth-data-labeling`. This parent bukcet will contain folders for different labeling tasks which in turn will contain folders for their annotations and other files. The reason is because we want each labeling task to have their own self-contained folder.

```bash
ground-truth-data-labeling 
|-- bounding_box_v1 # labeling task 1
    |-- ground_truth_annots # folder containing ground truth annotations
    |-- images # folder containing unannotated images
    |-- yolo_annot_files # folder containing yolo annotations
```

For our first labeling task, we will create a folder called `bounding_box_v1`. This folder will contain the following subfolders:
- `images` - This is where we will upload our unannotated images
- `ground_truth_annots` - This dir. is initially empty and the Ground Truth job will populate it with the ground truth annotations
- `yolo_annot_files` - This dir. is empty as well but we will write a script that will transform the ground truth annotations into yolo format.


We will use the function [`upload_images_to_s3`](../src/sagemaker/utils/utils_s3.py#:~:text=upload_images_to_s3) in the [`utils_s3.py`](../src/sagemaker/utils/utils_s3.py) file to upload our unannotated images to S3.


----------------------------------
<a name="2-creating-the-manifest-file"></a>
## 2. Creating the manifest file
A Ground Truth job requires a manifest file that contains the paths to the images that need to be labeled. The manifest file is a JSON file that contains a list of objects, each with a `source-ref` key that points to the image in S3. It is easy to create this manifest file when we have only a couple of images but most of the time we will be dealing with thousands to hundreds of thousands of images. Hence, we will write a script that will create the manifest file for us.


We can have a separate ```input.json``` file that will contain the configuration for the Ground Truth job. Four our case, we will use the ```config.yaml``` file to store the configuration under the `ground_truth` section.

```yaml
# Ground Truth Configuration
ground_truth:
  s3_bucket: "ground-truth-data-labeling"
  job_id: "bounding_box"
  ground_truth_job_name: "yolo-bbox"
  yolo_output_dir: "yolo_annot_files"
```

The function ```create_manifest``` in ```process_manifest.py``` creates a list of images in the image folder and then create the manifest file which we upload to S3 using the ```upload_manifest``` function. Note that we are reading the S3 bucket and job names from the ```config.yaml``` file. 

Now our S3 bucket should look like this:

```bash
ground-truth-data-labeling 
|-- bounding_box
    |-- ground_truth_annots (s3://cyudhist-ground-truth-data-labeling/bounding_box_v1/ground_truth_annots)
    |-- images (s3://cyudhist-ground-truth-data-labeling/bounding_box_v1/images)
    |-- yolo_annot_files (s3://cyudhist-ground-truth-data-labeling/bounding_box_v1/yolo_annot_files)
    |-- manifest.json (s3://cyudhist-ground-truth-data-labeling/bounding_box_v1/manifest.json)
```

This is a sample of the manifest file where we have 3 images:

```json
# manifest.json

{"source-ref": "s3://ground-truth-data-labeling/bounding_box/images/000000000139.jpg"}
{"source-ref": "s3://ground-truth-data-labeling/bounding_box/images/000000000785.jpg"}
{"source-ref": "s3://ground-truth-data-labeling/bounding_box/images/000000000872.jpg"}
```

----------------------------------
<a name="3-create-sagemaker-ground-truth-job"></a>
## 3. Create SageMaker Ground Truth job

Now we can start by creating a Ground Truth job by going to: [https://aws.amazon.com/es/sagemaker-ai/groundtruth/](https://aws.amazon.com/es/sagemaker-ai/groundtruth/). We go to Amazon SageMaker AI -> Labeling Jobs and click on `"Create labeling job"`.


For the Job Name, we will use `yolo-bbox-v1` as for each labeling job we will have a different version. Then we fill the input and output dataset locations as shown below:


Input dataset location:
Provide a path to the S3 location where your manifest file is stored.

```bash
s3://cyudhist-ground-truth-data-labeling/bounding_box_v1/manifest.json
```


Output dataset location:
Provide a path to the S3 location where you want your labeled dataset to be stored.

```bash
s3://cyudhist-ground-truth-data-labeling/bounding_box_v1/ground_truth_annots
```

We create a new IAM role and specify our specific S3 bucket: `ground-truth-data-labeling`.


![SageMaker Ground Truth Job Creation](../assets/Screenshot%202025-08-20%20095515.png)


In the task type, we select `Image` as Task Selection and `Bounding Box` as Task Type.

![SageMaker Ground Truth Job Creation](../assets/Screenshot%202025-08-20%20095553.png)


We create a team of labelers and provide the email addresses of the labelers. The labelers will receive an email with a link to the labeling job and their credentials.


![SageMaker Ground Truth Job Creation](../assets/Screenshot%202025-08-20%20095845.png)


We now need to create our label in the Bounding box labeling tool section and give instructions to our labelers on how to label the images. We can also upload bad  and good examples of labeling to help our labelers. Note that it is important that our labels are consistent else the model will not be able to learn properly.


![SageMaker Ground Truth Job Creation](../assets/Screenshot%202025-08-20%20095907.png)

We can also click on `Preview` to see how the labeling tool looks like.

![SageMaker Ground Truth Job Creation](../assets/Screenshot%202025-08-18%20105133.png)


We finish by clicking `Create job`.




----------------------------------
<a name="4-start-labeling"></a>
## 4. Start Labeling

We can now start the labeling job by clicking `Start job`.


![SageMaker Ground Truth Job Creation](../assets/Screenshot%202025-08-18%20105608.png)



After the labeling is complete, the status of the labeling job changes to `Complete` and a new JSON file called `output.manifest` containing the annotations appears at `s3://cyudhist-ground-truth-data-labeling/bounding_box/ground_truth_annots/yolo-bbox/manifests/output/output.manifest`.


----------------------------------
<a name="5-parse-the-annotations"></a>
## 5. Parse the annotations

After we have our labeled images, the Ground Truth job will populate the `ground_truth_annots` folder with the ground truth annotations. Most importantly, it will output a `output.manifest` file under `s3://ground-truth-data-labeling/bounding_box_v1/ground_truth_annots/yolo-bbox/manifests/output/output.manifest` as mentioned above.

```json
# output.manifest

{
  "source-ref": "s3://ground-truth-data-labeling/bounding_box_v1/images/000000000139.jpg",
  "yolo-bbox": {
    "image_size": [{"width": 640, "height": 426, "depth": 3}],
    "annotations": [
      {"class_id": 0, "top": 156, "left": 416, "height": 150, "width": 63},
      {"class_id": 0, "top": 161, "left": 378, "height": 56, "width": 29}
    ]
  },
  "yolo-bbox-metadata": {
    "objects": [{"confidence": 0}, {"confidence": 0}],
    "class-map": {"0": "person"},
    "type": "groundtruth/object-detection",
    "human-annotated": "yes",
    "creation-date": "2025-08-18T17:57:40.723894",
    "job-name": "labeling-job/yolo-bbox"
  }
}

{
  "source-ref": "s3://ground-truth-data-labeling/bounding_box_v1/images/000000000785.jpg",
  "yolo-bbox": {
    "image_size": [{"width": 640, "height": 425, "depth": 3}],
    "annotations": [
      {"class_id": 0, "top": 24, "left": 274, "height": 375, "width": 256}
    ]
  },
  "yolo-bbox-metadata": {
    "objects": [{"confidence": 0}],
    "class-map": {"0": "person"},
    "type": "groundtruth/object-detection",
    "human-annotated": "yes",
    "creation-date": "2025-08-18T17:57:40.726437",
    "job-name": "labeling-job/yolo-bbox"
  }
}

{
  "source-ref": "s3://ground-truth-data-labeling/bounding_box_v1/images/000000000872.jpg",
  "yolo-bbox": {
    "image_size": [{"width": 621, "height": 640, "depth": 3}],
    "annotations": [
      {"class_id": 0, "top": 132, "left": 158, "height": 488, "width": 193},
      {"class_id": 0, "top": 94, "left": 197, "height": 472, "width": 261}
    ]
  },
  "yolo-bbox-metadata": {
    "objects": [{"confidence": 0}, {"confidence": 0}],
    "class-map": {"0": "person"},
    "type": "groundtruth/object-detection",
    "human-annotated": "yes",
    "creation-date": "2025-08-18T17:57:40.725659",
    "job-name": "labeling-job/yolo-bbox"
  }
}
```

The `output.manifest` file shows the bounding box annotations for detected objects in an image from AWS SageMaker Ground Truth. For image `000000000139.jpg`, we have 2 person detections with different bounding boxes.

Each annotation object contains:
- `class_id`: 0 - Object class (0 = "person" based on the class-map)
- `top`: 156 - Y-coordinate of box's top edge (pixels from top of image)
- `left`: 416 - X-coordinate of box's left edge (pixels from left of image)
- `height`: 150 - Box height in pixels
- `width`: 63 - Box width in pixels

The coordinates use the standard top-left origin format where (0,0) is the top-left corner of the image.

We will use the function [`parse_gt_output`](../src/sagemaker/ground_truth/parsing_annotations.py#:~:text=parse_gt_output) in the [`parsing_annotations.py`](../src/sagemaker/ground_truth/parsing_annotations.py) file to parse the annotations and create a pandas dataframe. The function will read the `output.manifest` file from S3 and create a dataframe with the following columns:

```csv
# annot.csv

| img_file             | category | box_left | box_top | box_height | box_width | img_width | img_height |
|----------------------|----------|----------|---------|------------|-----------|-----------|------------|
| 000000000139.jpg     | person   | 416      | 156     | 150        | 63        | 640       | 426        |
| 000000000139.jpg     | person   | 378      | 161     | 56         | 29        | 640       | 426        |
| 000000000785.jpg     | person   | 274      | 24      | 375        | 256       | 640       | 425        |
| 000000000872.jpg     | person   | 158      | 132     | 488        | 193       | 621       | 640        |
| 000000000872.jpg     | person   | 197      | 94      | 472        | 261       | 621       | 640        |
```

Now we did not have to go through that step of pre-provessing the annotations and saving them as csv. We could directly have it transformed into YOLO format. However, at some point we may be required to get the raw annotations and do some analysis on them like checking the distribution of the annotations, distributions of image sizes, distribution of categories, etc.

This `annot.csv` file will be saved to S3 under `s3://ground-truth-data-labeling/bounding_box_v1/ground_truth_annots/yolo-bbox/annot.csv`.

----------------------------------
<a name="6-transform-the-annotations-into-yolo-format"></a>
## 6. Transform the annotations into YOLO format

 We are finally going to transform the our annotations into YOLO format which is in the format `class` `x_center` `y_center` `width` `height`. 

*Box coordinates must be in normalized xywh format (from 0 to 1). If your boxes are in pixels, you should divide x_center and width by image width, and y_center and height by image height. Class numbers should be zero-indexed (start with 0).* - [YOLO format](https://docs.ultralytics.com/datasets/detect/#ultralytics-yolo-format)

We will use the function [`process_yolo_format`](../src/sagemaker/ground_truth/create_yolo_annotations.py#:~:text=process_yolo_format) in the [`create_yolo_annotations.py`](../src/sagemaker/ground_truth/create_yolo_annotations.py) file to transform the annotations into YOLO format. The function will read the `annot.csv` file from S3 and create a dataframe which we will then save to S3 as a text file in the `yolo_annot_files` folder as `.txt` files using the function [`save_yolo_annotations_to_s3`](../src/sagemaker/ground_truth/create_yolo_annotations.py#:~:text=save_yolo_annotations_to_s3).


Below is the math that the function performs to transform the annotations into YOLO format.

```python
# Convert the category to an integer
df_ann["int_category"] = df_ann["category"].apply(lambda x: categories.index(x))

# Calculate the center of the box
df_ann["box_center_w"] = df_ann["box_left"] + df_ann["box_width"] / 2 # we divide by 2 because we want the center of the box
df_ann["box_center_h"] = df_ann["box_top"] + df_ann["box_height"] / 2

# scale box dimensions by image dimensions
df_ann["box_center_w"] = df_ann["box_center_w"] / df_ann["img_width"]
df_ann["box_center_h"] = df_ann["box_center_h"] / df_ann["img_height"]
df_ann["box_width"] = df_ann["box_width"] / df_ann["img_width"]
df_ann["box_height"] = df_ann["box_height"] / df_ann["img_height"]
```

This is an example of the output in YOLO format for the image `000000000139.jpg`.

```txt
# yolo_annot_files/000000000139.txt

0 0.6992 0.5423 0.0984 0.3521
0 0.6133 0.4437 0.0453 0.1315
```

The `.txt` file will be saved to S3 under `s3://ground-truth-data-labeling/bounding_box_v1/yolo_annot_files/`.







----------------------------------
<a name="7-visualize-yolo-annotations"></a>
## 7. Visualize YOLO Annotations

We can visualize the YOLO annotations locally. We first download the annotated image and their corresponding yolo annotations using the function [`download_yolo_annotations`](../src/sagemaker/utils/utils_s3.py#:~:text=download_yolo_annotations) in the [`utils_s3.py`](../src/sagemaker/utils/utils_s3.py) file and then use the function [`visualize_bbox`](../src/sagemaker/ground_truth/visualize_yolo_annotations.py#:~:text=visualize_bbox) in the [`visualize_yolo_annotations.py`](../src/sagemaker/ground_truth/visualize_yolo_annotations.py) file to visualize the bounding boxes.

This is an example of the output:



![Visualize YOLO Annotations](../assets/Screenshot%202025-08-18%20150459.png)


Note that it is important we get the order of the labels. It should match the order we used while creating the Ground Truth labeling job. If we don't get the order right, the labels will be assigned incorrectly. We can get this from `data.json` under the path: `s3://ground-truth-data-labeling/bounding_box_v1/ground_truth_annots/yolo-bbox/annotation-tool/data.json`.


```json
{"document-version":"2021-05-13","labels":[{"label":"person"}]}
```




----------------------------------

## Reference

[1] Amazon Web Services. (2023). "Streamlining data labeling for YOLO object detection in Amazon SageMaker Ground Truth." *AWS Machine Learning Blog*. Available at: https://aws.amazon.com/blogs/machine-learning/streamlining-data-labeling-for-yolo-object-detection-in-amazon-sagemaker-ground-truth/

[2] Amazon Web Services. (2023). "Create high-quality instructions for Amazon SageMaker Ground Truth labeling jobs." *AWS Machine Learning Blog*. Available at: https://aws.amazon.com/blogs/machine-learning/create-high-quality-instructions-for-amazon-sagemaker-ground-truth-labeling-jobs/







