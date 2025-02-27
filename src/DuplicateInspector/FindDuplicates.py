import os
from pathlib import Path
from tqdm import tqdm
import skimage.metrics
import skimage.io
# import imagehash

# Number of images to compare
nComparison = 5

# Textures directory
BASE_PATH = "E:/AI/Training/LoRAs/Foxes"


files = list(os.listdir())
files = [f for f in files if os.path.isfile(f) and Path(f).suffix in ['.jpg', '.png']]



currentImages = [None] * nComparison
matches = files.copy()  # We're going to modify the list so keep the original!
nDuplicates = 0
for n, i in tqdm(enumerate(files)):
    try:
        img = skimage.io.imread(i, as_gray=True)
        img = skimage.transform.resize(img, (256, 256), anti_aliasing=True)
    except Exception as e:
        print(f"Could not read image file {i} - {e}")
    for c in currentImages:
        if c is None:
            break
        try:
            res = skimage.metrics.normalized_root_mse(img, c[1])
            if res < 10e-12:
                nDuplicates += 1
                matches[n] = matches[c[0]]
                # print(f"Found duplicated image: {i}" )
                break
        except ValueError:
            pass # Image dimensions mismatch
    currentImages = [(n, img)] + currentImages[:-1]



print(f"Duplicates: {nDuplicates} / {len(files)}")

print(len(set(matches)))
print(len(files))
print(files)
print(matches)

