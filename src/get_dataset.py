
import fiftyone as fo
import os


def download_dataset(split: str, classes: list[str], max_samples: int) -> fo.Dataset:
    dataset = fo.zoo.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
    )
    return dataset


def main():
    train_dataset = download_dataset("train", ["person"], 10)
    val_dataset = download_dataset("validation", ["person"], 10)


    # download dataset in this dir
    # make dir if not exists
    if not os.path.exists("dataset/train"):
        os.makedirs("dataset/train")
    if not os.path.exists("dataset/validation"):
        os.makedirs("dataset/validation")

    train_dataset.export(
        export_dir="dataset/train",
        dataset_type=fo.types.COCODetectionDataset,
    )

    val_dataset.export(
        export_dir="dataset/validation",
        dataset_type=fo.types.COCODetectionDataset,
    )


if __name__ == "__main__":
    main()
    print("Dataset downloaded and exported successfully.")