import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import boto3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_s3 import download_yolo_annotations


def visualize_bbox(img_file: str, yolo_ann_file: str, label_dict: dict, figure_size: tuple = (6, 8)) -> None:
    """
    Plots bounding boxes on images

    Input:
    img_file : numpy.array
    yolo_ann_file: Text file containing annotations in YOLO format
    label_dict: Dictionary of image categories
    figure_size: Figure size
    """

    img = mpimg.imread(img_file)
    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    ax.imshow(img)

    im_height, im_width, _ = img.shape

    palette = mcolors.TABLEAU_COLORS
    colors = [c for c in palette.keys()]
    with open(yolo_ann_file, "r") as fin:
        for line in fin:
            cat, center_w, center_h, width, height = line.split()
            cat = int(cat)
            category_name = label_dict[cat]
            left = (float(center_w) - float(width) / 2) * im_width
            top = (float(center_h) - float(height) / 2) * im_height
            width = float(width) * im_width
            height = float(height) * im_height

            rect = plt.Rectangle(
                (left, top),
                width,
                height,
                fill=False,
                linewidth=2,
                edgecolor=colors[cat],
            )
            ax.add_patch(rect)
            props = dict(boxstyle="round", facecolor=colors[cat], alpha=0.5)
            ax.text(
                left,
                top,
                category_name,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            ) 
    plt.show()


def main():
    """
    Plots bounding boxes
    """
    # download the image and annotation file
    # s3_client = boto3.client("s3")
    # annotation_path, image_path = download_yolo_annotations(
    #     s3_client=s3_client,
    #     bucket="ground-truth-data-labeling",
    #     base_filenames="000000000139"
    # )

    # visualize the bounding boxes
    labels = {0: "person"}
    image_path = "src/sagemaker/sagemaker_data_processing/tmp/000000000139.jpg"
    annotation_path = "src/sagemaker/sagemaker_data_processing/tmp/000000000139.txt"
    visualize_bbox(image_path, annotation_path, labels)


if __name__ == "__main__":
    main()