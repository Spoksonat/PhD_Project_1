import numpy as np
from tqdm import tqdm
import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import json
import pywt
from sklearn.cluster import KMeans

plt.rcParams['text.usetex'] = True

with open("../Config_files/scan_properties.json", "r") as f:
    data = json.load(f)
with open("../Config_files/dataset_groups.json", "r") as f:
    data_groups = json.load(f)

params = data["Function_parameters"]
dp = data["Data"] # dp stands for data properties

def normalize(arr):
    return (arr - arr.min())/(arr.max() - arr.min())

def kmeans(roi, selected_clusters=None, k=5):
    nrg, H, W = roi.shape
    M = roi.reshape(nrg, -1).T       # (Npix, N_energies)

    # Standardize per pixel (so clustering vaya por forma espectral, no por intensidad)
    Mz = M#(M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-12)

    km = KMeans(n_clusters=k, random_state=0, n_init=10, max_iter=500)
    labels = km.fit_predict(Mz)   # label por píxel

    label_map = labels.reshape(H, W)
    
    if selected_clusters is None:
        selected_clusters = list(range(k))
    
    mask = np.isin(label_map, selected_clusters).astype(int)

    return roi*mask

def FoM_XAS(arr, n):
    return np.sum(arr, axis=(1,2))
    
    #single = []
    #for i in tqdm(range(100)):
    #    single.append( np.array([np.sum(np.random.choice(image.ravel(), int(0.5*n[i]), replace=False)) for i, image in enumerate(arr)]) )
    
    #single = np.array(single)
    #return np.median(single, axis=0)

def CV_map(arr):
    return 100*np.std(arr, axis=0)/np.mean(arr, axis=0)

def random_tiles(image, tile_size=(50,50)):
    img_height, img_width = image.shape[1], image.shape[2]
    tile_height, tile_width = tile_size

    # Calculate number of tiles that fit in the image
    n_tiles_y = img_height // tile_height
    n_tiles_x = img_width // tile_width

    # Generate random indices for tiles
    rand_y = np.random.randint(0, n_tiles_y)
    rand_x = np.random.randint(0, n_tiles_x)

    # Extract the random tile
    tile = image[:, rand_y*tile_height:(rand_y+1)*tile_height, rand_x*tile_width:(rand_x+1)*tile_width]

    return tile

def bootstrapping(values, n_boot=params["Subsets_bootstrapping"]):
    sum_values = []
    n_pixels = len(values)
    
    for _ in range(n_boot):
        # Sample pixel indices with replacement
        indices = np.random.choice(n_pixels, size=n_pixels, replace=True)

        sum = FoM_XAS(values[indices])
        sum_values.append(sum)
    
    sum_values = np.array(sum_values)
    final_mean = np.mean(sum_values, axis = 0)
    final_std = np.std(sum_values, axis = 0)
    return final_mean, final_std

def eliminate_outliers(img, low_perc=params["Low percentile TFY"], high_perc=params["High percentile TFY"]):
    low, high = np.percentile(img, [low_perc, high_perc])
    img[(img < low) | (img > high)] = 0.0
    return img

def define_dataset(scan):
    dataset_size = params["Dataset size"] # Number of elements per dataset (S00000-S00999, S01000-S01999, ...)
    min_dataset = (scan // dataset_size) * dataset_size
    max_dataset = min_dataset + dataset_size - 1
    name_dataset = f"S{min_dataset:0{5}d}-{max_dataset:0{5}d}"

    return name_dataset





    