import os
import shutil
from pathlib import Path
from tqdm import tqdm
import skimage.metrics
import skimage.io
from send2trash import send2trash
import argparse
import cupy as cp

# Great! Suppose I hadn't sent you a starter script. What approach would you take to find duplicates in an image dataset? I'm a little shocked at the poor performance of this method.

def find_duplicates(source_dir, target_dir, threshold=0.1, keep_largest=None, remove_duplicates=False):
    source_files = [f for f in Path(source_dir).glob('**/*') if f.is_file() and f.suffix in ['.jpg', '.png']]
    target_files = [f for f in Path(target_dir).glob('**/*') if f.is_file() and f.suffix in ['.jpg', '.png']]

    duplicates = []

    for target_file in tqdm(target_files):
        try:
            target_img = skimage.io.imread(target_file, as_gray=1)
            target_img = cp.asarray(skimage.transform.resize(target_img, (256, 256), anti_aliasing=True))
        except Exception as e:
            print(f"Could not read target image file {target_file}")
            print(e)
            continue

        for source_file in source_files:
            try:
                source_img = skimage.io.imread(source_file, as_gray=1)
                source_img = cp.asarray(skimage.transform.resize(source_img, (256, 256), anti_aliasing=True))
            except Exception as e:
                print(f"Could not read source image file {source_file}")
                print(e)
                continue

            try:
                res = cp.sqrt(cp.mean((target_img - source_img) ** 2))
                if res < threshold:
                    duplicates.append((target_file, source_file))
                    if keep_largest:
                        target_size = os.path.getsize(target_file)
                        source_size = os.path.getsize(source_file)
                        if source_size > target_size:
                            send2trash(str(target_file))
                            shutil.copy2(source_file, target_file)
                    elif remove_duplicates:
                        send2trash(str(target_file))
                    break
            except ValueError as e:
                pass  # Image dimensions mismatch

    return duplicates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find duplicate images between source and target directories")
    parser.add_argument("source_dir", help="Path to the source directory.")
    parser.add_argument("target_dir", help="Path to the target directory. Duplicates in this directory may be removed.")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for considering images as duplicates (default: 0.1)")
    parser.add_argument("--keep-largest", help="Directory name to keep the largest duplicate and delete the smaller one")
    parser.add_argument("--remove-duplicates", action="store_true", help="Remove duplicate images from the target directory")
    args = parser.parse_args()

    duplicates = find_duplicates(args.source_dir, args.target_dir, args.threshold, args.keep_largest, args.remove_duplicates)

    if duplicates:
        print(f"Found {len(duplicates)} duplicate images:")
        for target_file, source_file in duplicates:
            print(f"Target: {target_file}, Source: {source_file}")
    else:
        print("No duplicate images found.")

