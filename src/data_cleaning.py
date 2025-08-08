import json
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


from consts import TRAIN_LABELS_PATH, VAL_LABELS_PATH, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH


def load_coco_labels_as_dataframe(labels_json_path: str) -> pd.DataFrame:
    """Load COCO labels.json file and return as DataFrame with image names."""
    with open(labels_json_path, 'r') as f:
        data = json.load(f)
    
    # Get annotations as DataFrame
    df = pd.DataFrame(data['annotations'])
    
    # Create image_id to file_name mapping from images data
    images_df = pd.DataFrame(data['images'])
    image_mapping = images_df.set_index('id')['file_name'].to_dict()
    
    # Add image_name column by mapping image_id to file_name
    df['image_name'] = df['image_id'].map(image_mapping)
    
    # Separate bbox into individual columns
    df['bbox_x'] = df['bbox'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['bbox_y'] = df['bbox'].apply(lambda x: x[1] if len(x) > 1 else None)
    df['bbox_width'] = df['bbox'].apply(lambda x: x[2] if len(x) > 2 else None)
    df['bbox_height'] = df['bbox'].apply(lambda x: x[3] if len(x) > 3 else None)

    # filter by 'person' only
    df = df[df['supercategory'] == 'person']

    
    return df


def draw_bbox_on_images(data_path: str, images_path: str, grid_size: tuple = (2, 5), name: str = "image_label"):
    """
    Draw bounding boxes on images in a grid layout.
    
    Args:
        df: DataFrame with columns ['image_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']
        images_path: Path to the folder containing images
        grid_size: Tuple of (rows, cols) for the grid layout
    """

    # read csv
    df = pd.read_csv(data_path)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()
    
    # Group annotations by image
    grouped = df.groupby('image_name')
    
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    
    # Flatten axes array for easier indexing
    if rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, image_file in enumerate(image_files):
        if i >= rows * cols:
            break
            
        image_path = os.path.join(images_path, image_file)
        image = Image.open(image_path)
        
        axes[i].imshow(image)
        axes[i].set_title(image_file, fontsize=8)
        axes[i].axis('off')
        
        # Draw bounding boxes if annotations exist for this image
        if image_file in grouped.groups:
            annotations = grouped.get_group(image_file)
            for _, annotation in annotations.iterrows():
                x = annotation['bbox_x']
                y = annotation['bbox_y']
                w = annotation['bbox_width']
                h = annotation['bbox_height']
                
                # Create rectangle patch
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                axes[i].add_patch(rect)
                
                # Add label
                axes[i].text(x, y - 5, 'person', fontsize=8, color='red', weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Hide any unused subplots
    for i in range(len(image_files), rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save grid in the images directory
    plt.savefig(f"{name}.png", bbox_inches='tight', dpi=150)
    print(f"Bounding box grid saved: {name}.png")
    plt.close()



def display_images_grid(images_path: str, grid_size: tuple = (2, 5)):
    """
    Display all images in a folder as a grid.
    
    Args:
        images_path: Path to the folder containing images
        grid_size: Tuple of (rows, cols) for the grid layout
    """
    # Get all image files
    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # Sort for consistent order
    
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
    
    # Flatten axes array for easier indexing
    if rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, image_file in enumerate(image_files):
        if i >= rows * cols:
            break
            
        image_path = os.path.join(images_path, image_file)
        image = Image.open(image_path)
        
        axes[i].imshow(image)
        axes[i].set_title(image_file, fontsize=8)
        axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(image_files), rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_grid.png', bbox_inches='tight', dpi=150)
    print("Image grid saved as 'image_grid.png'")
    plt.close()


if __name__ == "__main__":
    df_train = load_coco_labels_as_dataframe(TRAIN_LABELS_PATH)
    df_val = load_coco_labels_as_dataframe(VAL_LABELS_PATH)
    print(df_train.head())
    print(df_val.head())
    print("Columns of training data: ", df_train.columns.tolist())
    print("Columns of validation data: ", df_val.columns.tolist())
    print("Size of training data: ", df_train.shape)
    print("Size of validation data: ", df_val.shape)
    print("Number of 'image_id' in training data: ", df_train['image_id'].nunique())
    print("Number of 'image_id' in validation data: ", df_val['image_id'].nunique())
    print("Number of 'category_id' in training data: ", df_train['category_id'].nunique())
    print("Number of 'category_id' in validation data: ", df_val['category_id'].nunique())
    print("Number of 'id' in training data: ", df_train['id'].nunique())
    print("Number of 'id' in validation data: ", df_val['id'].nunique())

    # drop columns that are not needed
    df_train = df_train.drop(columns=['id', 'image_id', 'category_id', 'iscrowd', 'area'])
    df_val = df_val.drop(columns=['id', 'image_id', 'category_id', 'iscrowd', 'area'])

    print(df_train.head())
    print(df_val.head())

    # export to csv
    df_train.to_csv("dataset/train/train_data_cleaned.csv", index=False)
    df_val.to_csv("dataset/validation/val_data_cleaned.csv", index=False)  

    # #     # Simple grid display of all images
    # print("\n=== Displaying Training Images Grid ===")
    # display_images_grid(TRAIN_IMAGES_PATH)
    
    # print("\n=== Displaying Validation Images Grid ===")
    # display_images_grid(VAL_IMAGES_PATH)
    
    # Draw bounding boxes in grid format
    draw_bbox_on_images("dataset/train/train_data_cleaned.csv", TRAIN_IMAGES_PATH, name="train_image_label")
    draw_bbox_on_images("dataset/validation/val_data_cleaned.csv", VAL_IMAGES_PATH, name="val_image_label")
