# %%
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import sunpy.map
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from p_tqdm import p_map

img_size_dic = {"MDI": 1024, "HMI": 4096}

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"

local_path_root = os.path.join(data_folder, dataset_folder)

df = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_name))
df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
times = Time(df["time"], format="jd")
df["datetime"] = pd.to_datetime(times.iso)

selected_df = df[~df["filtered"]]

lon_trshld = 70
front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]

min_size = 0.024

cleaned_df = front_df.copy()
for idx, row in cleaned_df.iterrows():
    x_min, y_min = row["bottom_left_cutout"]
    x_max, y_max = row["top_right_cutout"]

    img_sz = img_size_dic.get(row["instrument"])
    width = (x_max - x_min) / img_sz
    height = (y_max - y_min) / img_sz

    cleaned_df.at[idx, "width"] = width
    cleaned_df.at[idx, "height"] = height

cleaned_df = cleaned_df[(cleaned_df["width"] >= min_size) & (cleaned_df["height"] >= min_size)]

cleaned_df

# %%
base_dir = "/ARCAFF/data/cutouts_from_fulldisk"


# %%
def preprocess_FD(row):
    arccnet_path_root = row["path"].split("/fits")[0]
    image_path = row["path"].replace(arccnet_path_root, local_path_root)

    with fits.open(image_path) as img_fit:
        data = img_fit[1].data
        header = img_fit[1].header

    sunpy_map = sunpy.map.Map(data, header)

    # Generate a grid of coordinates for each pixel
    x, y = np.meshgrid(np.arange(sunpy_map.data.shape[1]), np.arange(sunpy_map.data.shape[0]))
    coordinates = sunpy_map.pixel_to_world(x * u.pix, y * u.pix)

    # Check if the coordinates are on the solar disk
    on_disk = coordinates.separation(sunpy_map.reference_coordinate) <= sunpy.map.solar_angular_radius(coordinates)

    # Mask data that is outside the solar disk
    sunpy_map.data[~on_disk] = np.nan  # Set off-disk pixels to NaN

    crota2 = sunpy_map.meta.get("CROTA2", 0)
    rotated_map = sunpy_map.rotate(angle=-crota2 * u.deg)

    return rotated_map


# %%
def process_row(row):
    filename = (
        f"{row['instrument']}_{row['datetime'].strftime('%Y%m%d_%H%M%S')}_NOAA_{row['NOAA']}_{row['magnetic_class']}"
    )
    save_dir = os.path.join(base_dir, "quicklook", row["instrument"])
    os.makedirs(save_dir, exist_ok=True)

    # Extract cutout dimensions
    img_size_dic.get(row["instrument"])
    x_min, y_min = row["bottom_left_cutout"]
    x_max, y_max = row["top_right_cutout"]
    width = x_max - x_min
    height = y_max - y_min

    # Process the map and cutout
    map = preprocess_FD(row)
    lower_left_coord = map.pixel_to_world(x_min * u.pix, y_min * u.pix)
    upper_right_coord = map.pixel_to_world(x_max * u.pix, y_max * u.pix)
    cutout_map = map.submap(bottom_left=lower_left_coord, top_right=upper_right_coord)

    # save cutout as fits file
    fits_path = os.path.join(base_dir, "fits", f"{filename}.fits")
    header = fits.Header()
    header['MAGCLASS'] = row['magnetic_class']  # Ensure the key is <= 8 characters
    header['NOAA'] = row['NOAA']
    fits.writeto(fits_path, cutout_map.data, header, overwrite=True)

    # Plot and save quicklook
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection=map)
    map.plot(axes=ax1, cmap="hmimag")
    map.draw_grid(color="white", alpha=0.25, axes=ax1)
    rect = Rectangle(
        (x_min, y_min),
        width,
        height,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        transform=ax1.get_transform("pixel"),
    )
    ax1.add_patch(rect)

    ax2 = fig.add_subplot(1, 2, 2, projection=cutout_map)
    cutout_map.plot(axes=ax2, cmap="hmimag")
    ax2.set_title(f"NOAA: {row['NOAA']} - {row['magnetic_class']}")

    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)

    image_array = cutout_map.data

    return image_array, row["magnetic_class"]


# %%
results = p_map(process_row, [row for _, row in cleaned_df.iterrows()])
image_arrays, labels = zip(*results)
, 

np.savez_compressed(os.path.join(base_dir, "dataset.npz"), 
                    image_arrays=np.array(image_arrays), labels=np.array(labels))
cleaned_df.to_parquet(os.path.join(base_dir,"dataframe.parquet"))