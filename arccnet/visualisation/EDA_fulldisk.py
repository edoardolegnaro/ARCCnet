# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ARCAFF
#     language: python
#     name: python3
# ---

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import sunpy.map
from matplotlib.patches import Rectangle
from scipy.ndimage import rotate

import astropy.units as u
# %%
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from arccnet.models import labels
from arccnet.visualisation import utils as ut_v

sns.set_style("darkgrid")
pd.set_option("display.max_columns", None)

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"

tab = Table.read(os.path.join(data_folder, dataset_folder, df_name))
df = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_name))
df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
times = Time(df["time"], format="jd")
df["datetime"] = pd.to_datetime(times.iso)

# %% [markdown]
# ## Classes Counts

# %%
discarded_df = df[df["filtered"]]
selected_df = df[~df["filtered"]]


# Calculate the percentage of selected items and unique fulldisks
selected_percentage = len(selected_df) / len(df) * 100
unique_fulldisks_count = len(selected_df["path"].unique())
unique_fulldisks_total = len(df["path"].unique())
unique_fulldisks_percentage = unique_fulldisks_count / unique_fulldisks_total * 100

# Print the formatted output
print(f"             Selected ARs: {len(selected_df)} out of {len(df)} " f"({selected_percentage:.2f}%)")
print(
    f"Selected Unique Fulldisks: {unique_fulldisks_count} out of {unique_fulldisks_total}  "
    f"({unique_fulldisks_percentage:.2f}%)"
)

# %%
discarded_counts = discarded_df["magnetic_class"].value_counts().sort_index()
selected_counts = selected_df["magnetic_class"].value_counts().sort_index()

all_classes = sorted(set(discarded_counts.index) | set(selected_counts.index))
discarded_counts = discarded_counts.reindex(all_classes, fill_value=0)
selected_counts = selected_counts.reindex(all_classes, fill_value=0)

counts_df = pd.DataFrame({"Selected": selected_counts, "Discarded": discarded_counts}, index=all_classes)

greek_labels = labels.convert_to_greek_label(all_classes)
class_to_greek = dict(zip(all_classes, greek_labels))
counts_df.index = counts_df.index.map(class_to_greek)

counts_df["Total"] = counts_df["Selected"] + counts_df["Discarded"]
grand_total = counts_df["Total"].sum()
counts_df["Selected_pct"] = counts_df["Selected"] / grand_total * 100
counts_df["Discarded_pct"] = counts_df["Discarded"] / grand_total * 100
counts_df["Total_pct"] = counts_df["Total"] / grand_total * 100

counts_df["Selected_formatted"] = counts_df.apply(lambda x: f"{int(x['Selected'])} ({x['Selected_pct']:.2f}%)", axis=1)
counts_df["Discarded_formatted"] = counts_df.apply(
    lambda x: f"{int(x['Discarded'])} ({x['Discarded_pct']:.2f}%)", axis=1
)
counts_df["Total_formatted"] = counts_df.apply(lambda x: f"{int(x['Total'])} ({x['Total_pct']:.2f}%)", axis=1)

formatted_counts_df = counts_df[["Selected_formatted", "Discarded_formatted", "Total_formatted"]]
formatted_counts_df.columns = ["Selected", "Discarded", "Total"]

formatted_counts_df

# %%
ut_v.make_classes_histogram(
    selected_df["magnetic_class"],
    y_off=20,
    figsz=(9, 5),
    title="FullDisk Dataset (non-filtered)",
    ylim=3400,
    transparent=True,
)
plt.show()

# %% [markdown]
# ## Time Distribution

# %%
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=selected_df[selected_df["instrument"] == "MDI"]["datetime"],
        y=[1] * len(selected_df[selected_df["instrument"] == "MDI"]),  # Assign a fixed y-value to bring them closer
        mode="markers",
        name="MDI",
        marker=dict(color="blue"),
    )
)

fig.add_trace(
    go.Scatter(
        x=selected_df[selected_df["instrument"] == "HMI"]["datetime"],
        y=[1.1] * len(selected_df[selected_df["instrument"] == "HMI"]),  # Slightly higher y-value, to separate them
        mode="markers",
        name="HMI",
        marker=dict(color="red"),
    )
)

fig.update_layout(
    title="Images over Time",
    xaxis_title="Date",
    yaxis_title="Instrument",
    yaxis=dict(
        tickvals=[1, 1.1],
        ticktext=["MDI", "HMI"],
        range=[0.95, 1.15],  # Adjust the range to make them closer
        showgrid=False,
    ),
    showlegend=True,  # Ensure legend shows up only once per instrument
)

# Show the figure
fig.show()

# %%
fig_ars = ut_v.months_years_heatmap(selected_df, "datetime", "Number of ARs per Month and Year", "Number of ARs")
fig_ars.show()

fig_fulldisks = ut_v.months_years_heatmap(
    selected_df.drop_duplicates(subset="path"), "datetime", "Number of Fulldisks", "Number of FDs"
)
fig_fulldisks.show()


# %% [markdown]
# ## Location on the Sun


# %%
def location_on_sun(df, pnt_color="#1f77b4"):
    # Convert latitude and longitude to radians
    latV = np.deg2rad(df["latitude"])
    lonV = np.deg2rad(df["longitude"])

    # Convert to y and z coordinates for the plot
    yV = np.cos(latV) * np.sin(lonV)
    zV = np.sin(latV)

    # Prepare the solar disc as a circle
    theta = np.linspace(0, 2 * np.pi, 100)
    solar_disc_y = np.cos(theta)
    solar_disc_z = np.sin(theta)

    # Extract additional information for hover text (index and time)
    hover_text = [f"Index: {i}<br>Time: {time}" for i, time in zip(range(len(selected_df)), selected_df["datetime"])]

    # Create the interactive plot
    fig = go.Figure()

    # Add the solar disc (as a circular boundary)
    fig.add_trace(
        go.Scatter(x=solar_disc_y, y=solar_disc_z, mode="lines", line=dict(color="gray", width=1), showlegend=False)
    )

    # Add ARs locations with hover info
    fig.add_trace(
        go.Scatter(
            x=yV,
            y=zV,
            mode="markers",
            marker=dict(size=3, color=pnt_color, opacity=0.7),
            text=hover_text,  # Custom hover text (index and time)
            hoverinfo="text",  # Display custom text in hover
            showlegend=False,
        )
    )

    # Add meridians and parallels
    num_meridians = 12
    num_parallels = 12
    num_points = 300

    phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)  # Angles for meridians
    lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)  # Latitude angles (parallels)

    # Angles from south to north pole
    theta_meridian = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Plot each meridian
    for phi in phis:
        y_meridian = np.cos(theta_meridian) * np.sin(phi)
        z_meridian = np.sin(theta_meridian)
        fig.add_trace(
            go.Scatter(x=y_meridian, y=z_meridian, mode="lines", line=dict(color="black", width=0.2), showlegend=False)
        )

    # Plot each parallel
    for lat in lats:
        radius = np.cos(lat)  # This defines the radius of the latitude circle in the y-z plane
        y_parallel = radius * np.sin(theta_meridian)
        z_parallel = np.sin(lat) * np.ones(num_points)
        fig.add_trace(
            go.Scatter(x=y_parallel, y=z_parallel, mode="lines", line=dict(color="black", width=0.2), showlegend=False)
        )

    # Update the layout
    fig.update_layout(
        title="ARs Location on the Sun",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        xaxis_range=[-1.1, 1.1],
        yaxis_range=[-1.1, 1.1],
        width=800,
        height=800,
        autosize=False,
        hovermode="closest",
        margin=dict(l=50, r=50, b=50, t=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Set equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


sun_location_fig = location_on_sun(selected_df)
sun_location_fig.show()

# %%
bins = np.arange(-90, 91, 15)
x_ticks = np.arange(-90, 91, 15)
x_tick_labels = [f"{int(x)}°" for x in x_ticks]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].hist(selected_df["longitude"], bins=bins, edgecolor="black")
axes[0].set_xticks(x_ticks)
axes[0].set_xticklabels(x_tick_labels)
axes[0].set_xlabel("Degrees")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Longitude")

axes[1].hist(selected_df["latitude"], bins=bins, edgecolor="black")
axes[1].set_xticks(x_ticks)
axes[1].set_xticklabels(x_tick_labels)
axes[1].set_xlabel("Degrees")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Latitude")

# Adjust layout
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Magnetogram Labels

# %%
arccnet_path_root = "arccnet_data/02_intermediate/data/mag"
local_path_root = os.path.join(data_folder, dataset_folder)

# %%
idx = 18

row = selected_df.iloc[idx]

image_path = row["path"].replace(arccnet_path_root, local_path_root)
image_labels = selected_df[selected_df["path"] == row["path"]]

fig, ax = plt.subplots(figsize=(10, 10))

with fits.open(image_path) as img_fit:
    header = img_fit[1].header
    data = np.array(img_fit[1].data, dtype=float)
    data = np.nan_to_num(data, nan=0.0)
    data = ut_v.hardtanh_transform_npy(data)
    crota2 = header["CROTA2"]
    data = rotate(data, crota2, reshape=False, mode="constant", cval=0)  # `reshape=False` to maintain image size
    ax.imshow(data, origin="lower", cmap=ut_v.magnetic_map, vmin=-1, vmax=1)
    for _, label_row in image_labels.iterrows():
        x_min, y_min = label_row["bottom_left_cutout"]
        x_max, y_max = label_row["top_right_cutout"]

        width = x_max - x_min
        height = y_max - y_min

        rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none")  # (x, y)

        ax.add_patch(rect)

        label_text = label_row["magnetic_class"]
        center_x = (x_min + x_max) / 2

        ax.text(
            center_x,
            y_max + 5,
            label_text,
            color="white",
            fontsize=10,
            ha="center",
            va="bottom",
            bbox=dict(facecolor="black", alpha=0.5),
        )
    ax.set_title(row["datetime"])
plt.show()

# %%
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
sunpy_map

# %%
# Extract CROTA2 value from the header
crota2 = sunpy_map.meta.get("CROTA2", 0)  # Default to 0 if CROTA2 is not present

# Apply the rotation based on the CROTA2 value
rotated_map = sunpy_map.rotate(angle=-crota2 * u.deg)  # Apply the rotation, -CROTA2 to align it correctly

# Plot the rotated map
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection=rotated_map)

# Plot the map with the adjusted rotation
rotated_map.plot(axes=ax, cmap="hmimag")

# Draw grid if needed
rotated_map.draw_grid(axes=ax)

for _, label_row in image_labels.iterrows():
    x_min, y_min = label_row["bottom_left_cutout"]
    x_max, y_max = label_row["top_right_cutout"]

    width = x_max - x_min
    height = y_max - y_min

    rect = Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor="red", facecolor="none")  # (x, y)

    ax.add_patch(rect)

    label_text = label_row["magnetic_class"]
    center_x = (x_min + x_max) / 2

    ax.text(
        center_x,
        y_max + 5,
        label_text,
        color="white",
        fontsize=10,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="black", alpha=0.5),
    )

plt.show()

# %% [markdown]
# ## Cutout Sizes

# %%
ut_v.w_h_scatterplot(selected_df)

# %%
widths, heights, _, _ = ut_v.compute_widths_heights(selected_df)

aspect_ratios = np.array(widths) / np.array(heights)

hist_fig = go.Figure()

# Add a histogram trace for aspect ratios
hist_fig.add_trace(go.Histogram(x=aspect_ratios, nbinsx=50, marker=dict(color="blue"), opacity=0.75))

# Update the layout for the histogram
hist_fig.update_layout(
    title="Aspect Ratios", xaxis_title="Aspect Ratio (Width/Height)", yaxis_title="Count", autosize=True
)

# Show the histogram
hist_fig.show()

# %% [markdown]
# # Clean up dataset

# %%
lon_trshld = 70
front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]
back_df = selected_df[(selected_df["longitude"] >= lon_trshld) | (selected_df["longitude"] <= -lon_trshld)]

# %%
front_fig = ut_v.location_on_sun(front_df)
combined_fig = ut_v.location_on_sun(back_df, fig=front_fig, color="darkorange")
combined_fig.show()

# %%
ut_v.w_h_scatterplot(front_df)

# %%
min_size = 0.024
img_size_dic = {"MDI": 1024, "HMI": 4096}

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

# %%
ut_v.plot_fd(cleaned_df.iloc[1627], cleaned_df, local_path_root)

# %%
ut_v.w_h_scatterplot(cleaned_df)
