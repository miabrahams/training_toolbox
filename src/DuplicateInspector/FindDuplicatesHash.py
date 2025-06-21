
import os
import imagehash
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil
from send2trash import send2trash
import pickle



def generate_image_pairs_html(duplicates_pairs, start_index, end_index):

    grid_items = ''

    for image_path1, image_path2 in tqdm(sorted(duplicates_pairs)[start_index:end_index]):

        grid_items += f'''
            <div class="image-cell">
                <h3>{os.path.basename(image_path1)}</h3>
                <img src="file:///{image_path1}">
            </div>
            <div class="image-cell">
                <h3>{os.path.basename(image_path2)}</h3>
                <img src="file:///{image_path2}">
            </div>
        '''

    style = '''
.image-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
}

img {
    max-height: 400px;
    width: auto;
    object-fit: contain;
}
'''


    html = f'''<!DOCTYPE html>
<html>
<head>
<style>
{style}
</style>
</head>
    <body style="width: 100%; display: flex; justify-content: center; background-color: #f0f0f0;">
        <div style="display: grid; grid-template-columns: repeat(2, auto); gap: 20px; margin: 20px">
            {grid_items}
        </div>
    </body>
</html>
    '''
    return html



def valid_files(directory):
    return [f for f in Path(directory).glob('**/*') if f.is_file() and f.suffix in ['.jpg', '.png']]

def cached_hash(source_dir, hash_size, output_file):
    source_files = valid_files(source_dir)
    print(f"Found {len(source_files)} images.")
    if len(source_files) == 0:
        return None

    hashes = {}

    if os.path.exists(output_file):
        print("Found cached hashes. Loading...")
        hashes = pickle.load(open(output_file, "rb"))
        # Remove files that no longer exist
        hashes = {filename: file_hash for filename, file_hash in hashes.items() if os.path.exists(os.path.join(source_dir, filename))}

        # Ensure we've cached the hash of the right length
        if hashes and len(hashes) > 0:
            cached_hash_size = len(list(hashes.values())[0])
            if cached_hash_size != hash_size ** 2:
                print("Hash size mismatch. Recomputing hashes.")
                hashes = {}

    for source_file in tqdm([f for f in source_files if f not in hashes], desc="Hashing source images"):
        try:
            with Image.open(source_file) as img:
                hash_value = imagehash.phash(img, hash_size=hash_size)
                hashes[source_file] = hash_value
        except Exception as e:
            print(f"Could not hash image file {source_file}. Error: {e}")

    pickle.dump(hashes, open(output_file, "wb"))
    return hashes

# Keep only largest file
def handle_keep_largest(path1, path2, different_dirs=False):
    size1 = os.path.getsize(path1)
    size2 = os.path.getsize(path2)
    if size1 > size2:
        print(f"Deleting file {path2}")
        send2trash(str(path2))
        if different_dirs:
            shutil.copy2(path1, path2)
    else:
        print(f"Deleting file {path1}")
        send2trash(str(path1))
        if different_dirs:
            shutil.copy2(path2, path1)



def find_duplicates_single_dir(directory, hash_size, max_distance, keep_largest):
    hashes = cached_hash(directory, hash_size, os.path.join(directory, "hashes.pkl"))

    if hashes is None:
        print("No comparisons possible.")
        return []

    duplicates = []
    files = list(hashes.items())

    for i in tqdm(range(len(files)), desc="Comparing images"):
        for j in range(i + 1, len(files)):
            file1, hash1 = files[i]
            file2, hash2 = files[j]

            if hash1 - hash2 <= max_distance:
                print(f"Found duplicate images. Keep largest: {keep_largest}")
                fullpath1 = os.path.join(directory, file1)
                fullpath2 = os.path.join(directory, file2)
                duplicates.append((fullpath1, fullpath2))

                if keep_largest:
                    handle_keep_largest(fullpath1, fullpath2)
                # Only find one duplicate per image
                break

    return duplicates

def find_duplicates_alternate_dir(source_dir, alternates_dir, hash_size, max_distance, keep_largest, remove_alternates):
    source_hashes = cached_hash(source_dir, hash_size, os.path.join(source_dir, "hashes.pkl"))
    alternate_hashes = cached_hash(alternates_dir, hash_size, os.path.join(alternates_dir, "hashes.pkl"))

    if source_hashes is None or alternate_hashes is None:
        print("No comparisons possible.")
        return []

    duplicates = []
    for alternate_file, alternate_hash in tqdm(alternate_hashes.items(), desc="Comparing alternate images"):
        alternate_fullpath = os.path.join(alternates_dir, alternate_file)
        try:
            for source_file, source_hash in source_hashes.items():
                source_fullpath = os.path.join(source_dir, source_file)
                if alternate_hash - source_hash <= max_distance:
                    duplicates.append((alternate_fullpath, source_fullpath))
                    if keep_largest:
                        handle_keep_largest(alternate_fullpath, source_fullpath, True)
                    elif remove_alternates:
                        send2trash(str(alternate_fullpath))
                    break
        except Exception as e:
            print(f"Could not read alternates image file {alternate_fullpath}. Error: {e}")

    return duplicates

def find_duplicates(source_dir, alternates_dir, hash_size, max_distance, keep_largest, remove_alternates):
    if alternates_dir is None or alternates_dir == source_dir:
        duplicates = find_duplicates_single_dir(source_dir, hash_size, max_distance, keep_largest)
    else:
        duplicates = find_duplicates_alternate_dir(source_dir, alternates_dir, hash_size, max_distance, keep_largest, remove_alternates)

    if duplicates:
        print(f"Found {len(duplicates)} duplicate images:")
        for alternate_file, source_file in duplicates:
            print(f"Source: {source_file},  Alternate: {alternate_file}")
        if args.write_html:
            html = generate_image_pairs_html(duplicates, 0, len(duplicates))
            html = html.replace("/mnt/d/", "D:/")
            with open('data/duplicates.html', 'w') as file:
                file.write(html)
    else:
        print("No duplicate images found.")

    return duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find duplicate images between source and alternates directories")
    parser.add_argument("source_dir", help="Path to the source directory.")
    parser.add_argument("--alternates-dir", default=None, help="Path to the alternates directory. Duplicates in this directory may be removed.")
    parser.add_argument("--hash-size", type=int, default=12, help="Image Hash size to compute(default: 8)")
    parser.add_argument("--max-distance", type=int, default=0, help="Hamming-Distance threshold for considering images as alternates (default: 0)")
    parser.add_argument("--keep-largest", action="store_true", help="If enabled, remove the smaller duplicate and copy the larger one to the other directory.")
    parser.add_argument("--remove-alternates", action="store_true", help="Remove duplicate images from the alternates directory. Only used if keep-largest is not enabled.")
    parser.add_argument("--write-html", action="store_true", help="Write an HTML file with the duplicate image pairs.")
    args = parser.parse_args()



    duplicates = find_duplicates(args.source_dir, args.alternates_dir, args.hash_size, args.max_distance, args.keep_largest, args.remove_alternates)
