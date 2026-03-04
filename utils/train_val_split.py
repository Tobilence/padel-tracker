import os
import random
import shutil
from pathlib import Path


def prepare_yolo_dataset(
    data_dir="data",
    train_ratio=0.8,
    seed=42,
    image_extensions=(".jpg", ".jpeg", ".png")
):
    """
    Rearranges dataset into:

    data/
        images/
            train/
            val/
        labels/
            train/
            val/

    Assumes:
        images and labels currently live in:
            data/images/
            data/labels/

        label filename == image filename (.txt)
    """

    random.seed(seed)

    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    # New folders
    img_train = images_dir / "train"
    img_val = images_dir / "val"
    lbl_train = labels_dir / "train"
    lbl_val = labels_dir / "val"

    # Create directories
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect images (ignore already moved ones)
    images = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if len(images) == 0:
        print("No images found.")
        return

    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")

    def move_files(image_list, img_dest, lbl_dest):
        for img_path in image_list:
            label_path = labels_dir / f"{img_path.stem}.txt"

            # Move image
            shutil.move(str(img_path), img_dest / img_path.name)

            # Move label if exists
            if label_path.exists():
                shutil.move(str(label_path), lbl_dest / label_path.name)
            else:
                print(f"⚠ Missing label for {img_path.name}")

    move_files(train_images, img_train, lbl_train)
    move_files(val_images, img_val, lbl_val)

    print("\n✅ Dataset successfully prepared!")


if __name__ == "__main__":
    prepare_yolo_dataset(
        data_dir="../experiments/exp2/data",
        train_ratio=0.8  
    )