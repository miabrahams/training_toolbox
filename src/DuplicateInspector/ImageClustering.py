# Script to cluster images nonparametrically.
# Useful to categorize artists and their effects on image generation.
import os
from PIL import Image
import pickle
from tqdm import tqdm

# SVD method
import matplotlib.pyplot as plt
import numpy as np


# scikit-learn faces demo
from sklearn import cluster, decomposition, metrics


BASE_PATH = "E:/AI/stable-diffusion-webui/outputs/txt2img-grids/crosskemonoFurryModel_crosskemono20/Artist_Experiments_Neutral"
os.chdir(BASE_PATH)


# Extract the artist from prompt
def artist_from_metadata(params):
    # format: (by ARTIST:1.25)
    # Open parentheses always in the same location.
    closeparen = params.find(")")
    artist = params[31:(closeparen - 5)]
    return artist

# Create a list of images in each subfolder
imagegrids = {}
for subfolder in os.listdir():
    if os.path.isdir(subfolder) and (subfolder.find('-') > -1):
        for image in os.listdir(subfolder):
            im = Image.open(os.path.join(subfolder, image))
            im.load()
            params = im.info['parameters']
            artist = artist_from_metadata(params)
            imagegrids[artist] = im
            print(artist)
pickle.dump(imagegrids, open("imagegrids.pkl", "wb"))




##########################################
### Load images
##########################################

imagegrids = pickle.load(open("imagegrids.pkl", "rb"))
artists = list(imagegrids.keys())

# Prepare dimensions on test image
test_img = imagegrids[artists[0]]
orig_size = test_img.size
crop_size = (0, 0, orig_size[0] / 2, orig_size[1])
test_img.crop(crop_size)
new_size = [round(x / 4) for x in test_img.size]  # 448x144, 224x72


# Compile images into ndarray
img_data = np.empty((0, new_size[0] * new_size[1]))
for artist in tqdm(artists):
    # Read image, convert to grayscale and resize
    img = imagegrids[artist]
    img = img.convert('L').crop(crop_size).resize(new_size)
    # Convert the image to a Numpy array and flatten it into a vector
    img_vector = np.asarray(img).flatten() / 255.0
    # Add the flattened image vector to the img_vectors array as a new row
    img_data = np.vstack((img_data, img_vector))


# Print the shape of the array (should be num_images, img_size[0] * img_size[1]))
print(img_data.shape)
n_samples, n_features = img_data.shape




##########################################
### SVD method.
##########################################


# Global centering (focus on one feature, centering all samples)
img_mean = np.mean(img_data, 0)
img_centered = (img_data - img_mean)

# Local centering (focus on one sample, centering all features)
img_centered = img_centered - img_centered.mean(axis=1).reshape(n_samples, -1)


# Plotting function
n_row, n_col = 5, 3
n_components = n_row * n_col
image_size = new_size[1], new_size[0]
def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.colormaps['gray']):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(3.0 * n_col, 1.2 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    fig.suptitle(title, size=16)
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_size),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")
    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()

plot_gallery("Images from dataset", img_centered[:n_components])


# Full PCA
pca = decomposition.PCA(whiten=True).fit(img_centered)
print(np.cumsum(pca.explained_variance_ratio_))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# 100 dimensions explains about 80% variance


# Reduced PCA
pca = decomposition.PCA(100, whiten=True).fit(img_centered)
converted_data = pca.fit_transform(np.array(img_centered))
converted_data.shape


#### kmeans algorithm
### Choosing best Num cluster for KMeans
for NumbOfcluster in range(2, 5):
    kmean = cluster.KMeans(n_clusters=NumbOfcluster, max_iter=2000)
    kmean.fit(img_centered)
    labels = kmean.labels_
    KMeans_Sil = metrics.silhouette_score(converted_data, kmean.labels_, metric='euclidean')
    print('Kmeans silhouette ',KMeans_Sil)


kmean = cluster.KMeans(init="k-means++", n_clusters=10, n_init=4)
fit = kmean.fit(X = converted_data)

for seed in range(5):
    kmeans = cluster.KMeans(n_clusters=10, max_iter=10000, n_init=3, random_state=seed).fit(converted_data)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    sizes = list(cluster_sizes)
    sizes.sort()
    print(f"Number of elements asigned to each cluster: {sizes}")

# Few components
pca_estimator = decomposition.PCA(
    n_components=n_components, svd_solver="randomized", whiten=True
)
pca_estimator.fit(img_centered)
plot_gallery(
    "Eigenimages - PCA using randomized SVD",
     pca_estimator.components_[:n_components]
)

pca_estimator.singular_values_





