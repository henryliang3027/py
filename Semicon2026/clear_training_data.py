#!/usr/bin/env python3
"""Clear auto-labeled training data written by image_receive_gen_data.py.

Deletes all files under:
    /py/for_training/box/images/   /py/for_training/box/labels/
    /py/for_training/date/images/  /py/for_training/date/labels/

The four directories themselves are kept (only their contents are removed).
"""

import os
import sys

TRAINING_DIR = "/py/for_training"
TARGET_DIRS = [
    os.path.join(TRAINING_DIR, "box", "images"),
    os.path.join(TRAINING_DIR, "box", "labels"),
    os.path.join(TRAINING_DIR, "date", "images"),
    os.path.join(TRAINING_DIR, "date", "labels"),
]


def count_files(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for name in os.listdir(d) if os.path.isfile(os.path.join(d, name)))


def clear_dir(d):
    removed = 0
    for name in os.listdir(d):
        path = os.path.join(d, name)
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    return removed


def main():
    counts = {d: count_files(d) for d in TARGET_DIRS}
    total = sum(counts.values())

    if total == 0:
        print("Nothing to clear -- all target directories are already empty.")
        return

    print("About to delete:")
    for d in TARGET_DIRS:
        print(f"  {d}: {counts[d]} file(s)")

    if "--yes" not in sys.argv:
        reply = input(f"Delete {total} file(s) total? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return

    for d in TARGET_DIRS:
        if os.path.isdir(d):
            removed = clear_dir(d)
            print(f"  cleared {d}: {removed} file(s) removed")

    print("Done.")


if __name__ == "__main__":
    main()
