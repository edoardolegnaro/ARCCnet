"""Verify YOLO label distribution and consistency."""

from pathlib import Path
from collections import Counter


def check_labels(yolo_root: Path, split: str = "train"):
    """Check label distribution in YOLO dataset."""
    labels_dir = yolo_root / "labels" / split

    if not labels_dir.exists():
        print(f"Directory not found: {labels_dir}")
        return

    all_classes = []
    label_files = list(labels_dir.glob("*.txt"))

    for label_file in label_files:
        with open(label_file) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    all_classes.append(class_id)

    counter = Counter(all_classes)
    print(f"\n{split.upper()} set - Label distribution:")
    print(f"  Total annotations: {len(all_classes)}")
    print(f"  Total images: {len(label_files)}")
    print("  Class distribution:")
    for class_id in sorted(counter.keys()):
        count = counter[class_id]
        pct = count / len(all_classes) * 100
        print(f"    Class {class_id}: {count:5d} ({pct:5.1f}%)")

    return counter


if __name__ == "__main__":
    # Check both mag and cont datasets
    for data_type in ["mag", "cont"]:
        print(f"\n{'=' * 60}")
        print(f"{data_type.upper()} Dataset")
        print(f"{'=' * 60}")

        yolo_root = Path(f"/ARCAFF/data/YOLO/{data_type}")

        if yolo_root.exists():
            train_counter = check_labels(yolo_root, "train")
            val_counter = check_labels(yolo_root, "val")

            # Check for invalid class IDs
            all_classes = set()
            if train_counter:
                all_classes.update(train_counter.keys())
            if val_counter:
                all_classes.update(val_counter.keys())

            if all_classes:
                max_class = max(all_classes)
                min_class = min(all_classes)
                print(f"\n  Class ID range: {min_class} to {max_class}")
                if max_class > 2:
                    print("  ⚠️  WARNING: Found class IDs > 2 (expected 0-2)")
                if min_class < 0:
                    print("  ⚠️  WARNING: Found negative class IDs")
        else:
            print(f"  Dataset not found at {yolo_root}")
