import os
import json
from io import StringIO
import boto3
import s3fs
import pandas as pd


from utils.utils_config import load_ground_truth_config


def process_yolo_format(annot_file: str, categories: list) -> pd.DataFrame:
    """
    Prepares the annotation in YOLO format

    Input:
    annot_file: csv file containing Ground Truth annotations
    categories: List of object categories in proper ORDER for model training

    Returns:
    df_ann: pandas dataframe with the following columns
            img_file int_category box_center_w box_center_h box_width box_height


    Note:
    YOLO data format: <object-class> <x_center> <y_center> <width> <height>
    """

    df_ann = pd.read_csv(annot_file)

    # convert the category to an integer based on the order of the categories
    # Example:
    # categories = ["person", "car", "dog"]
    # df_ann["category"] = ["person", "dog", "car"]
    # df_ann["int_category"] = [0, 2, 1]
    df_ann["int_category"] = df_ann["category"].apply(lambda x: categories.index(x))

    # calculate the center of the box
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


def main():
    """
    Performs the following tasks:
    1. Collect the category names from the Ground Truth job
    2. Creates a dataframe with annotations in YOLO format
    3. Saves a text file in S3 with YOLO annotations for each of the labeled images
    """
    # load config
    config = load_ground_truth_config()
    s3_bucket = config["s3_bucket"]
    job_id = config["job_id"]
    gt_job_name = config["ground_truth_job_name"]
    yolo_output = config["yolo_output_dir"]

    # get categories
    s3_path_cats = (f"s3://{s3_bucket}/{job_id}/ground_truth_annots/{gt_job_name}/annotation-tool/data.json")
    categories = get_categories(s3_path_cats)
    print(f"\n labels used in Ground Truth job: {categories}\n")

    # get annotation file
    gt_annot_file = f"s3://{s3_bucket}/{job_id}/ground_truth_annots/{gt_job_name}/annot.csv"
    s3_dir = f"{job_id}/{yolo_output}"
    print(f"annotation files saved in: {s3_dir}\n")

    # process the annotations into YOLO format
    df_annot = process_yolo_format(gt_annot_file, categories)
    save_yolo_annotations_to_s3(s3_bucket, s3_dir, df_annot)


if __name__ == "__main__":
    main()