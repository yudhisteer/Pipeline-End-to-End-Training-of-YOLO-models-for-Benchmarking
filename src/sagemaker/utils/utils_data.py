"""
Utility functions for data management
"""

import json
from io import StringIO

import boto3
import pandas as pd
import s3fs


from utils.utils_config import load_ground_truth_config


def parse_gt_output(config: dict) -> pd.DataFrame:
    """
    Captures the json Ground Truth bounding box annotations into a pandas dataframe

    Input:
    config: dictionary containing the configuration for the Ground Truth job

    Returns:
    df_bbox: pandas dataframe with bounding box coordinates
             for each item in every image
    """
    # load config
    config = load_ground_truth_config()
    job_name = config["ground_truth_job_name"]
    manifest_path = config["output_manifest_file"]

    filesys = s3fs.S3FileSystem()
    with filesys.open(manifest_path) as fin:
        annot_list = []
        for line in fin.readlines():
            record = json.loads(line)
            if job_name in record.keys():
                # get image file path and name
                image_file_path = record["source-ref"]
                image_file_name = image_file_path.split("/")[-1]
                class_maps = record[f"{job_name}-metadata"]["class-map"]

                # get image size
                imsize_list = record[job_name]["image_size"]
                assert len(imsize_list) == 1
                image_width = imsize_list[0]["width"]
                image_height = imsize_list[0]["height"]

                # get annotations
                for annot in record[job_name]["annotations"]:
                    left = annot["left"]
                    top = annot["top"]
                    height = annot["height"]
                    width = annot["width"]
                    class_name = class_maps[f'{annot["class_id"]}']

                    annot_list.append(
                        [
                            image_file_name,
                            class_name,
                            left,
                            top,
                            height,
                            width,
                            image_width,
                            image_height,
                        ]
                    )

        df_bbox = pd.DataFrame(
            annot_list,
            columns=[
                "img_file",
                "category",
                "box_left",
                "box_top",
                "box_height",
                "box_width",
                "img_width",
                "img_height",
            ],
        )

    return df_bbox


def save_df_to_s3(df_local: pd.DataFrame, s3_bucket: str, destination: str):
    """
    Saves a pandas dataframe to S3

    Input:
    df_local: pandas dataframe to save
    s3_bucket: S3 bucket name
    destination: S3 destination path
    """
    # create a buffer
    csv_buffer = StringIO()
    s3_resource = boto3.resource("s3")

    # save the df to buffer
    df_local.to_csv(csv_buffer, index=False)
    s3_resource.Object(s3_bucket, destination).put(Body=csv_buffer.getvalue())


def process_yolo_format(
    annot_input: pd.DataFrame | str, categories: list,
) -> pd.DataFrame:
    """
    Prepares the annotation in YOLO format from either a DataFrame or CSV file path

    Input:
    annot_input: Either a pandas DataFrame or a string path to a CSV file containing Ground Truth annotations
                DataFrame should have columns: img_file, category, box_left, box_top, box_height, box_width, img_width, img_height
                CSV file should have the same columns
    categories: List of object categories in proper ORDER for model training

    Returns:
    df_ann: pandas dataframe with the following columns
            img_file int_category box_center_w box_center_h box_width box_height

    Note:
    YOLO data format: <object-class> <x_center> <y_center> <width> <height>
    """

    if isinstance(annot_input, str):
        # read the CSV
        df_ann = pd.read_csv(annot_input)
    elif isinstance(annot_input, pd.DataFrame):
        # already a DataFrame, use it directly
        df_ann = annot_input.copy()
    else:
        raise TypeError(
            "annot_input must be either a string (file path) or pandas DataFrame"
        )

    # Convert the category to an integer
    df_ann["int_category"] = df_ann["category"].apply(lambda x: categories.index(x))

    # Calculate the center of the box
    df_ann["box_center_w"] = df_ann["box_left"] + df_ann["box_width"] / 2
    df_ann["box_center_h"] = df_ann["box_top"] + df_ann["box_height"] / 2

    # scale box dimensions by image dimensions
    df_ann["box_center_w"] = df_ann["box_center_w"] / df_ann["img_width"]
    df_ann["box_center_h"] = df_ann["box_center_h"] / df_ann["img_height"]
    df_ann["box_width"] = df_ann["box_width"] / df_ann["img_width"]
    df_ann["box_height"] = df_ann["box_height"] / df_ann["img_height"]

    return df_ann


def save_yolo_annotations_to_s3(s3_bucket: str, prefix: str, df_local: pd.DataFrame):
    """
    For every image in the dataset, save a text file with annotation in YOLO format

    Example output format:
    s3://my-bucket/my-prefix/image1.txt
    s3://my-bucket/my-prefix/image2.txt
    ...

    Input:
    s3_bucket: S3 bucket name
    prefix: Folder name under s3_bucket where files will be written
    df_local: pandas dataframe with the following columns
              img_file int_category box_center_w box_center_h box_width box_height
    """

    unique_images = df_local["img_file"].unique()
    s3_resource = boto3.resource("s3")

    for image_file in unique_images:
        df_single_img_annots = df_local.loc[df_local.img_file == image_file]
        annot_txt_file = image_file.split(".")[0] + ".txt"
        destination = f"{prefix}/{annot_txt_file}"

        csv_buffer = StringIO()
        df_single_img_annots.to_csv(
            csv_buffer,
            index=False,
            header=False,
            sep=" ",
            float_format="%.4f",
            columns=[
                "int_category",
                "box_center_w",
                "box_center_h",
                "box_width",
                "box_height",
            ],
        )
        s3_resource.Object(s3_bucket, destination).put(Body=csv_buffer.getvalue())


def get_categories(json_file: str) -> list:
    """
    Makes a list of the category names in proper order

    Input:
    json_file: s3 path of the json file containing the category information

    Returns:
    cats: List of category names
    """

    filesys = s3fs.S3FileSystem()
    with filesys.open(json_file) as fin:
        line = fin.readline()
        record = json.loads(line)
        labels = [item["label"] for item in record["labels"]]

    return labels
