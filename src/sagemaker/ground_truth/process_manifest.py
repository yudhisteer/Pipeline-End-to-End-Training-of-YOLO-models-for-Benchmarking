import boto3
import json


from utils.utils_config import load_ground_truth_config


def create_manifest(config: dict) -> str:
    """
    Creates the manifest file for the Ground Truth job

    Input:
    config: Dictionary containing the ground truth configuration

    Returns:
    manifest_file: The manifest file required for GT job
    """

    # get s3 bucket and prefix from config
    s3_rec = boto3.resource("s3")
    s3_bucket = config["s3_bucket"]
    prefix = config["job_id"]
    image_folder = f"{prefix}/images"
    print(f"using images from ... {image_folder} \n")

    # get list of images in image folder
    bucket = s3_rec.Bucket(s3_bucket)
    objs = list(bucket.objects.filter(Prefix=image_folder))
    img_files = objs[1:]  # first item is the folder name
    n_imgs = len(img_files)
    print(f"there are {n_imgs} images \n")

    # create manifest file
    TOKEN = "source-ref"
    manifest_file = "/tmp/manifest.json"
    with open(manifest_file, "w") as fout:
        for img_file in img_files:
            fname = f"s3://{s3_bucket}/{img_file.key}"
            fout.write(f'{{"{TOKEN}": "{fname}"}}\n')

    return manifest_file


def upload_manifest(config: dict, manifest_file: str) -> None:
    """
    Uploads the manifest file into S3

    Input:
    config: Dictionary containing the ground truth configuration
    manifest_file: Path to the local copy of the manifest file
    """

    # get s3 bucket and prefix from config: TODO: maybe create a function for this to avoid code duplication
    s3_rec = boto3.resource("s3")
    s3_bucket = config["s3_bucket"]
    source = manifest_file.split("/")[-1]
    prefix = config["job_id"]
    destination = f"{prefix}/{source}"

    # upload the manifest file to s3
    print(f"uploading manifest file to {destination} \n")
    s3_rec.meta.client.upload_file(manifest_file, s3_bucket, destination)


def main():
    """
    Performs the following tasks:
    1. Collects image names from S3 and creates the manifest file for GT
    2. Uploads the manifest file to S3
    """

    # load ground truth config
    ground_truth_config = load_ground_truth_config()

    # create manifest file
    manifest_file = create_manifest(ground_truth_config)

    # upload manifest file to S3
    upload_manifest(ground_truth_config, manifest_file)


if __name__ == "__main__":
    main()