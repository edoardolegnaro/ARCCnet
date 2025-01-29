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

# %%
# %load_ext autoreload
# %autoreload 2

import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import display
from matplotlib import pyplot as plt
from p_tqdm import p_map
from scipy.stats import kurtosis, skew

from astropy.io import fits

import arccnet.models.cutouts.mcintosh.dataset_utils as mci_ut_d
from arccnet.models import dataset_utils as ut_d
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)


# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../data")
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_file_name = "cutout-mcintosh-catalog-v20240715.parq"

# %% [markdown]
# # McIntosh Classification

# %%
AR_df, encoders, mappings = mci_ut_d.process_ar_dataset(
    data_folder=data_folder,
    dataset_folder=dataset_folder,
    df_name=df_file_name,
    plot_histograms=True,
)


# %%
def group_and_sort_classes(class_list):
    # Group classes by their initial letter
    grouped_classes = defaultdict(list)
    for cls in sorted(class_list):  # Sort the entire list alphabetically first
        grouped_classes[cls[0]].append(cls)

    # Format the output
    for letter, classes in grouped_classes.items():
        print(f"{letter}: {', '.join(classes)}")


print("------ McIntosh Classes ------")
group_and_sort_classes(list(AR_df["mcintosh_class"].unique()))
print("\n------ Grouped McIntosh Classes ------")
group_and_sort_classes(
    list((AR_df["Z_component_grouped"] + AR_df["p_component_grouped"] + AR_df["c_component_grouped"]).unique())
)

# %%
ut_v.make_classes_histogram(
    AR_df["mcintosh_class"],
    horizontal=True,
    figsz=(10, 18),
    y_off=20,  # Horizontal text offset (from bar ends)
    x_rotation=0,
    ylabel="Number of Active Regions",
    title="McIntosh Class Distribution",
    ylim=4300,
)
plt.show()

# %%
ut_v.make_classes_histogram(
    AR_df["Z_component_grouped"] + AR_df["p_component_grouped"] + AR_df["c_component_grouped"],
    horizontal=True,
    figsz=(8, 12),
    y_off=20,  # Horizontal text offset (from bar ends)
    x_rotation=0,
    ylabel="Number of Active Regions",
    title="McIntosh Class Distribution",
    ylim=4700,
)
plt.show()

# %% [markdown]
# # Mount Wilson

# %%
df, AR_IA = ut_d.make_dataframe(data_folder, dataset_folder, df_file_name)

# %%
ut_v.make_classes_histogram(df["label"], y_off=20, figsz=(13, 6), title="Cutout Dataset")
plt.show()

# %%
ut_v.make_classes_histogram(AR_IA["label"], y_off=20, figsz=(10, 6), title="Cutout Dataset")
plt.show()

# %% [markdown]
# # Time Distribution

# %%
# Count ARs per day
ar_count_per_day = AR_IA["dates"].value_counts().sort_index()

# Create the histogram with Plotly
fig = px.bar(
    x=ar_count_per_day.index,
    y=ar_count_per_day.values,
    labels={"x": "Date", "y": "Number of ARs per Day"},
    title="ARs per Day",
    color_discrete_sequence=["black"],
).update_layout(plot_bgcolor="white")

# Show the plot
fig.show()


# %%
n_days = AR_IA["dates"].dt.date.nunique()

with plt.style.context("seaborn-v0_8-darkgrid"):
    plt.figure(figsize=(12, 6))
    plt.bar(ar_count_per_day.index, ar_count_per_day.values)
    plt.xlabel("Time")
    plt.ylabel("n° of ARs per day")
    plt.yticks(np.arange(0, 18 + 2, 2))
    plt.ylim([0, 17])
    plt.show()

# %%
# Subset the data for MDI and HMI
AR_df_MDI = AR_IA[AR_IA["quicklook_path_mdi"] != ""]
AR_df_HMI = AR_IA[AR_IA["quicklook_path_hmi"] != ""]

# Count ARs per day for MDI and HMI
mdi_count_per_day = AR_df_MDI["dates"].value_counts().sort_index()
hmi_count_per_day = AR_df_HMI["dates"].value_counts().sort_index()

time_counts_MDI = AR_df_MDI["dates"].value_counts().sort_index()
time_counts_HMI = AR_df_HMI["dates"].value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.scatter(time_counts_MDI.index, time_counts_MDI.values, color="b", alpha=0.1, label="MDI", s=2)
plt.scatter(time_counts_HMI.index, time_counts_HMI.values, color="r", alpha=0.1, label="HMI", s=2)
plt.xlabel("date", fontsize=11)
plt.ylabel("n° of ARs", fontsize=11)
plt.yticks([i for i in range(0, 20, 2)])
plt.legend(fontsize=11)
plt.show()

# %%
dates_MDI = AR_df_MDI["dates"].values
dates_HMI = AR_df_HMI["dates"].values
colors = ["blue", "red"]
labels = ["MDI", "HMI"]

plt.figure(figsize=(16, 2))

for idx, (dates_val, label) in enumerate(zip([dates_MDI, dates_HMI], labels)):
    # Create a scatter plot for each dataset with a label
    plt.scatter(dates_val, [idx] * len(dates_val), s=1, color=colors[idx], label=label)

# Format x-axis to show dates
plt.gca().xaxis_date()
plt.xticks(rotation=45)
plt.yticks([])  # Hide y-axis
plt.ylim([-1, 2])
plt.title("Dataset Timeline")

# Add legend
plt.legend(loc="upper left")  # Choose location as appropriate

plt.show()

# %% [markdown]
# # ARs Location on the Sun

# %%
lonV = np.deg2rad(np.where(AR_IA["processed_path_image_hmi"] != "", AR_IA["longitude_hmi"], AR_IA["longitude_mdi"]))
degree_ticks_lon = np.arange(-90, 91, 30)
rad_ticks_lon = np.deg2rad(degree_ticks_lon)

latV = np.deg2rad(np.where(AR_IA["processed_path_image_hmi"] != "", AR_IA["latitude_hmi"], AR_IA["latitude_mdi"]))
degree_ticks_lat = np.arange(-90, 91, 30)
rad_ticks_lat = np.deg2rad(degree_ticks_lat)

with sns.axes_style("darkgrid"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Longitude histogram (left)
    ax1.hist(lonV, bins=rad_ticks_lon, color="#4C72B0", edgecolor="black")
    ax1.set_xticks(rad_ticks_lon)
    ax1.set_xticklabels([f"{deg}°" for deg in degree_ticks_lon], rotation=0)
    ax1.set_xlabel("Longitude (degrees)")
    ax1.set_ylabel("Frequency")

    # Latitude histogram (right)
    ax2.hist(latV, bins=rad_ticks_lat, color="#4C72B0", edgecolor="black")
    ax2.set_xticks(rad_ticks_lat)
    ax2.set_xticklabels([f"{deg}°" for deg in degree_ticks_lat], rotation=0)
    ax2.set_xlabel("Latitude (degrees)")
    ax2.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# %%
long_limit_deg = 65

deg = np.pi / 180

yV = np.cos(latV) * np.sin(lonV)
zV = np.sin(latV)

condition = (lonV < -long_limit_deg * deg) | (lonV > long_limit_deg * deg)

AR_df_filtered = AR_IA[~condition]
AR_df_rear = AR_IA[condition]

rear_latV = latV[condition]
rear_lonV = lonV[condition]
rear_yV = yV[condition]
rear_zV = zV[condition]

front_latV = latV[~condition]
front_lonV = lonV[~condition]
front_yV = yV[~condition]
front_zV = zV[~condition]

# ARs' location on the solar disc
circle = plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none")
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(circle)

num_meridians = 12
num_parallels = 12
num_points = 300

# Angles for the meridians and parallels
phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)
lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)

# Angles from south to north pole
theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)

# Plot each meridian
for phi in phis:
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    ax.plot(y, z, "k-", linewidth=0.2)

# Plot each parallel
for lat in lats:
    radius = np.cos(lat)  # This defines the radius of the latitude circle in the y-z plane
    y = radius * np.sin(theta)
    z = np.sin(lat) * np.ones(num_points)
    ax.plot(y, z, "k-", linewidth=0.2)

ax.scatter(rear_yV, rear_zV, s=1, alpha=0.2, color="r", label="Rear")
ax.scatter(front_yV, front_zV, s=1, alpha=0.2, color="b", label="Front")

ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(fontsize=12)

plt.show()

print(f"Rear ARs: {len(rear_latV)}")
print(f"Front ARs: {len(front_latV)}")
print(f"Percentage of rear ARs: {100*len(rear_latV)/( len(rear_latV) + len(front_latV)):.2f}%")

# %%
ut_v.make_classes_histogram(AR_df_filtered["label"], title="Front ARs", y_off=10, figsz=(11, 5))
ut_v.make_classes_histogram(AR_df_rear["label"], title="Rear ARs", y_off=10, figsz=(11, 5))
plt.show()


# %% [markdown]
# # Statistical Analysis

# %% [markdown]
# ### Compute Statistics


# %%
def compute_statistics_for_row(row):
    """
    Computes statistics for a single row of the DataFrame.
    """
    results = {}
    path = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] != "" else "path_image_cutout_mdi"
    fits_file_path = os.path.join(data_folder, dataset_folder, row[path])

    with fits.open(fits_file_path) as img_fits:
        data = np.array(img_fits[1].data, dtype=float)
        data = np.nan_to_num(data, nan=0.0)
        results["min"] = np.min(data)
        results["max"] = np.max(data)
        results["mean"] = np.mean(data)
        results["std"] = np.std(data)
        results["var"] = np.var(data)
        results["median"] = np.median(data)
        results["skewness"] = skew(data.flatten())
        results["kurtosis"] = kurtosis(data.flatten())
        results["range"] = np.ptp(data)
        q75, q25 = np.percentile(data, [75, 25])
        results["iqr"] = q75 - q25

    return results


def compute_statistics(df, cache=False, cache_file=None):
    """
    Applies compute_statistics_for_row to each row in the DataFrame in parallel.
    If cache is True, it will load results from cache_file if it exists, otherwise compute and save.
    """
    if cache and cache_file and os.path.exists(cache_file):
        # Load cached results
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = p_map(compute_statistics_for_row, [row for _, row in df.iterrows()])

        # Save results to cache if caching is enabled
        if cache and cache_file:
            print(f"Saving results to cache file {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)

    stats_df = pd.DataFrame(results, index=df.index)

    return stats_df


title_mapping = {
    "min": "Min Values",
    "max": "Max Values",
    "mean": "Mean Values",
    "std": "Standard Deviations",
    "var": "Variances",
    "median": "Median Values",
    "iqr": "Interquartile Range (IQR) Values",
    "skewness": "Skewness Values",
    "kurtosis": "Kurtosis Values",
    "range": "Range Values",
}


def find_outliers(data):
    """
    Identifies outliers in a dataset using the interquartile range (IQR) method.

    Parameters:
    - data (array): Input data, which can be a list, array, or Pandas Series.

    Returns:
    - array: Indices of outliers in the input data array.

    If the input data contains non-numeric values, they are converted to NaN (Not a Number).
    The function then calculates the lower and upper bounds for outliers based on the IQR method.
    Data points falling below the lower bound or above the upper bound are considered outliers.
    The function returns the indices of these outlier values in the original input data array.
    """

    data_numeric = np.array(data, dtype=float)  # Converts non-convertible values to NaN
    nan_indices = np.where(np.isnan(data_numeric))[0]  # Find indices where values are NaN
    valid_data = data_numeric[~np.isnan(data_numeric)]  # Filter out NaNs to get valid numeric data

    if valid_data.size == 0:  # Check if valid_data has any elements left
        return nan_indices

    # Calculating Q1 and Q3 from the valid numeric data
    Q1 = np.percentile(valid_data, 25)
    Q3 = np.percentile(valid_data, 75)

    IQR = Q3 - Q1

    # Defining bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identifying outlier indices in the valid numeric data
    outlier_bool = (valid_data < lower_bound) | (valid_data > upper_bound)
    outlier_indices = np.where(outlier_bool)[0]

    # Convert indices in valid_data back to original indices (excluding NaNs)
    valid_indices = np.where(~np.isnan(data_numeric))[0]
    original_indices = valid_indices[outlier_indices]

    # Combine the indices of NaN and numeric outliers
    all_outlier_indices = np.concatenate((original_indices, nan_indices))

    return all_outlier_indices


# %%
df_alpha = df[df["magnetic_class"] == "Alpha"].reset_index(drop=False)
df_beta = df[df["magnetic_class"] == "Beta"].reset_index(drop=False)
df_betax = pd.concat(
    [
        df[df["magnetic_class"] == "Beta-Gamma"],
        df[df["magnetic_class"] == "Beta-Delta"],
        df[df["magnetic_class"] == "Beta-Gamma-Delta"],
    ]
).reset_index(drop=False)

# %%
cache_dir = os.path.join(os.getcwd(), "cache")
os.makedirs(cache_dir, exist_ok=True)
results_alpha = compute_statistics(df_alpha, cache=True, cache_file=os.path.join(cache_dir, "results_alpha_cache.pkl"))
results_beta = compute_statistics(df_beta, cache=True, cache_file=os.path.join(cache_dir, "results_beta_cache.pkl"))
results_betax = compute_statistics(df_betax, cache=True, cache_file=os.path.join(cache_dir, "results_betax_cache.pkl"))

# %%
df_alpha_results = pd.DataFrame(results_alpha)
df_beta_results = pd.DataFrame(results_beta)
df_betax_results = pd.DataFrame(results_betax)
df_alpha_results["Group"] = "Alpha"
df_beta_results["Group"] = "Beta"
df_betax_results["Group"] = "Beta-X"
df_results_combined = pd.concat([df_alpha_results, df_beta_results, df_betax_results], ignore_index=True)
combined_describe = df_results_combined.groupby("Group").describe()

# %% [markdown]
# ### Visualize Statistics

# %%
columns_to_display = 8
num_chunks = (combined_describe.shape[1] + columns_to_display - 1) // columns_to_display


def format_func(x):
    return f"{x:.2f}"


for i in range(num_chunks):
    start_col = i * columns_to_display
    end_col = start_col + columns_to_display
    display(combined_describe.iloc[:, start_col:end_col].style.format(format_func))


# %% [markdown]
# #### Histograms


# %%
def plot_histograms(key):
    # Calculate weights for each dataset so that the sum of the weights is 1
    weights_alpha = np.ones_like(results_alpha[key]) / len(results_alpha[key])
    weights_beta = np.ones_like(results_beta[key]) / len(results_beta[key])
    weights_betax = np.ones_like(results_betax[key]) / len(results_betax[key])
    # Plot histograms
    plt.hist(results_alpha[key], weights=weights_alpha, alpha=0.35, label="Alpha", density=True)
    plt.hist(results_beta[key], weights=weights_beta, alpha=0.35, label="Beta", density=True)
    plt.hist(results_betax[key], weights=weights_betax, alpha=0.35, label="Beta-X", density=True)
    plt.legend()
    plt.title("Normalized Histograms of " + title_mapping[key])
    plt.xlabel(key.title() + " Value")
    plt.ylabel("Relative Frequency")
    plt.show()


def plot_boxplots(data, labels, title):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels)
    plt.title(title)
    plt.ylabel("Value")
    plt.show()


for key in title_mapping:
    plot_histograms(key)


# %% [markdown]
# #### Violin Plots


# %%
# Combine the data into a single DataFrame
def results_to_df(results, label):
    df = pd.DataFrame(results)
    df["Group"] = label
    return df


combined_df = pd.concat(
    [results_to_df(results_alpha, "Alpha"), results_to_df(results_beta, "Beta"), results_to_df(results_betax, "Beta-x")]
)

if combined_df.isnull().values.any():
    combined_df = combined_df.dropna()

for stat in title_mapping:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Group", y=stat, data=combined_df, color="lightblue")
    plt.title(title_mapping[stat])
    plt.xlabel("Magnetic Class")
    plt.ylabel(f"{stat.capitalize()}")
    plt.show()

# %% [markdown]
# #### Box Plots

# %%
for key in title_mapping:
    plot_boxplots(
        [
            results_alpha[key][~np.isnan(results_alpha[key])],
            results_beta[key][~np.isnan(results_beta[key])],
            results_betax[key][~np.isnan(results_betax[key])],
        ],
        ["Alpha", "Beta", "Beta-x"],
        title_mapping[key],  # Use the formal title from the mapping
    )
