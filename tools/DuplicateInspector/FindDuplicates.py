import os
from pathlib import Path
from tqdm import tqdm
import skimage.metrics
import skimage.io
import skimage.transform
# import imagehash

from src.lib.config import get_settings

# Number of images to compare
nComparison = 5

settings = get_settings()
BASE_PATH = Path(settings.get("duplicate_inspector.base_path", ".")).expanduser().resolve()

files = [
    f for f in BASE_PATH.iterdir()
    if f.is_file() and f.suffix.lower() in ['.jpg', '.png']
]

currentImages = [None] * nComparison
matches = files.copy()  # We're going to modify the list so keep the original!
nDuplicates = 0
for n, image_path in tqdm(enumerate(files)):
    try:
        img = skimage.io.imread(str(image_path), as_gray=True)
        img = skimage.transform.resize(img, (256, 256), anti_aliasing=True)
    except Exception as e:
        print(f"Could not read image file {image_path} - {e}")
        continue
    for c in currentImages:
        if c is None:
            break
        try:
            res = skimage.metrics.normalized_root_mse(img, c[1])
            if res < 10e-12:
                nDuplicates += 1
                matches[n] = matches[c[0]]
                # print(f"Found duplicated image: {image_path}" )
                break
        except ValueError:
            pass # Image dimensions mismatch
    currentImages = [(n, img)] + currentImages[:-1]


print(f"Duplicates: {nDuplicates} / {len(files)}")

print(len(set(matches)))
print(len(files))
print(files)
print(matches)
