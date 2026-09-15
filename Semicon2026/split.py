"""Split YOLO-format datasets (box/date) into train/valid sets for Roboflow.

Reads images+labels from /py/for_training/<box|date>/{images,labels}
and writes an 80/20 train/valid split into /py/for_roboflow_split/<box|date>/{train,valid}/{images,labels}.
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

SRC_ROOT = Path(__file__).parent / "for_training"
DST_ROOT = Path(__file__).parent / "for_roboflow_split"
DATASETS = ["box", "date"]
LABEL_TEMPLATES = {
    "box": Path(__file__).parent / "label_template_box_11cls.txt",
    "date": Path(__file__).parent / "label_template_date_1cls.txt",
}


def split_dataset(name: str, train_ratio: float, seed: int) -> None:
    src_images = SRC_ROOT / name / "images"
    src_labels = SRC_ROOT / name / "labels"

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[{name}] no images found in {src_images}, skipping")
        return

    rng = random.Random(seed)
    rng.shuffle(images)

    split_idx = round(len(images) * train_ratio)
    splits = {"train": images[:split_idx], "val": images[split_idx:]}

    for split_name, split_images in splits.items():
        dst_images = DST_ROOT / name / "images" / split_name
        dst_labels = DST_ROOT / name / "labels" / split_name
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            label_path = src_labels / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dst_images / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, dst_labels / label_path.name)
            else:
                print(f"[{name}] warning: missing label for {image_path.name}")

        print(f"[{name}] {split_name}: {len(split_images)} images")


def zip_dataset(name: str) -> None:
    dataset_dir = DST_ROOT / name
    if not dataset_dir.exists():
        print(f"[{name}] nothing to zip, {dataset_dir} does not exist")
        return

    archive_path = DST_ROOT / f"{name}.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(dataset_dir))

        template_path = LABEL_TEMPLATES.get(name)
        if template_path and template_path.exists():
            zf.write(template_path, template_path.name)
        elif template_path:
            print(f"[{name}] warning: label template not found at {template_path}")

    print(f"[{name}] zipped to {archive_path}")


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--zip", type=str2bool, nargs="?", const=True, default=True,
        help="zip the box/ and date/ folders into for_roboflow_split/box.zip and date.zip",
    )
    args = parser.parse_args()

    for name in DATASETS:
        split_dataset(name, args.train_ratio, args.seed)

    if args.zip:
        for name in DATASETS:
            zip_dataset(name)


if __name__ == "__main__":
    main()
