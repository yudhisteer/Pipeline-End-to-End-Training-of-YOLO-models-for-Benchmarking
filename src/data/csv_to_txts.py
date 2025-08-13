# convert_csv_to_txt.py

import os
import pandas as pd
from PIL import Image
import shutil

def convert_csv_to_yolo(csv_path, images_dir, labels_dir, class_id=0):
    """
    Convert bounding boxes from CSV to YOLO format .txt files.
    
    Args:
        csv_path (str): Path to cleaned CSV file (e.g., train_data_cleaned.csv)
        images_dir (str): Path to folder with images
        labels_dir (str): Output folder for YOLO .txt label files
        class_id (int): YOLO class ID (default: 0 for 'person')
    """
    os.makedirs(labels_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Group by image_name so each file gets its own .txt
    grouped = df.groupby('image_name')

    # Get all image filenames in case some have no annotations
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in all_images:
        img_path = os.path.join(images_dir, image_file)
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Prepare .txt path
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")

        if image_file in grouped.groups:
            rows = grouped.get_group(image_file)
            yolo_lines = []
            for _, row in rows.iterrows():
                # CSV has bbox_x, bbox_y as top-left in pixels
                x_min = row['bbox_x']
                y_min = row['bbox_y']
                w = row['bbox_width']
                h = row['bbox_height']

                # Convert to YOLO normalized center format
                center_x = (x_min + w / 2) / img_w
                center_y = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

            # Write to file
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
        else:
            # No annotations: create empty file
            open(label_path, 'w').close()

    print(f"✅ Labels saved to {labels_dir}")

def rename_dataset_structure():
    val_old = "dataset/validation"
    val_new = "dataset/val"
    if os.path.exists(val_old):
        if os.path.exists(val_new):
            shutil.rmtree(val_new)
        os.rename(val_old, val_new)
        print(f"📂 Renamed {val_old} -> {val_new}")


    for split in ["dataset/train", "dataset/val"]:
        data_dir = os.path.join(split, "data")
        images_dir = os.path.join(split, "images")
        if os.path.exists(data_dir):
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
            os.rename(data_dir, images_dir)
            print(f"📂 Renamed {data_dir} -> {images_dir}")


if __name__ == "__main__":
    # Train
    convert_csv_to_yolo(
        csv_path="dataset/train/train_data_cleaned.csv",
        images_dir="dataset/train/data",
        labels_dir="dataset/train/labels",
        class_id=0
    )

    # Validation
    convert_csv_to_yolo(
        csv_path="dataset/validation/val_data_cleaned.csv",
        images_dir="dataset/validation/data",
        labels_dir="dataset/validation/labels",
        class_id=0
    )

    rename_dataset_structure()
