from io import StringIO
import json
import s3fs
import boto3
import pandas as pd

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
    # load the config
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
    # create buffer to store dataframe
    csv_buffer = StringIO()
    s3_resource = boto3.resource("s3")

    # save  dataframe to buffer
    df_local.to_csv(csv_buffer, index=False)
    s3_resource.Object(s3_bucket, destination).put(Body=csv_buffer.getvalue())


def main():
    """
    Performs the following tasks:
    1. Parses the Ground Truth annotations and creates a dataframe
    2. Saves the dataframe to S3
    """
    # load the config
    config = load_ground_truth_config()

    # parse the Ground Truth annotations and create dataframe
    df_annot = parse_gt_output(config)

    # save dataframe  S3
    destination = f"{config['job_id']}/ground_truth_annots/{config['ground_truth_job_name']}/annot.csv"
    save_df_to_s3(df_annot, config['s3_bucket'], destination)


if __name__ == "__main__":
    main()