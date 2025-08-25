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
#     display_name: py_3.11
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from astropy.io import fits

from arccnet import load_config
from arccnet.models import dataset_utils as ut_d
from arccnet.visualisation import utils as ut_v

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
config = load_config()

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
dataset_folder = "arccnet-v20250805/04_final"
df_file_name = "data/cutout_classification/region_classification.parq"
dataset_title = "arccnet v20250805"

# %%
df, _ = ut_d.make_dataframe(data_folder, dataset_folder, df_file_name)
ut_v.make_classes_histogram(df["label"], figsz=(18, 6), text_fontsize=11, title=dataset_title)
plt.show()
df

# %%
mdi_color = "royalblue"
hmi_color = "tomato"

df_MDI = df[df["path_image_cutout_hmi"] == ""].copy()
df_HMI = df[df["path_image_cutout_mdi"] == ""].copy()

# %% [markdown]
# # Quality Flags

# %%
QUALITY_FLAGS = {
    "SOHO/MDI": {
        0x00000001: "Missing Data",
        0x00000002: "Saturated Pixel",
        0x00000004: "Truncated (Top)",
        0x00000008: "Truncated (Bottom)",
        0x00000200: "Shutterless Mode",
        0x00010000: "Cosmic Ray",
        0x00020000: "Calibration Mode",
        0x00040000: "Image Bad",
    },
    "SDO/HMI": {
        0x00000020: "Missing >50% Data",
        0x00000080: "Limb Darkening Correction Bad",
        0x00000400: "Shutterless Mode",
        0x00001000: "Partial/Missing Frame",
        0x00010000: "Cosmic Ray",
    },
}


def decode_flags(flag_hex, flag_dict):
    """Decode hex flag to human-readable status."""
    try:
        flag_str = str(flag_hex).strip().lstrip("0x")
        if not flag_str or flag_str in ["nan", "None", "<NA>"]:
            return "Good Quality"
        flag_int = int(flag_str, 16)
        if flag_int == 0:
            return "Good Quality"
        meanings = [meaning for bit_val, meaning in flag_dict.items() if flag_int & bit_val]
        return " | ".join(meanings) or "Unknown Flag"
    except (ValueError, TypeError):
        return "Invalid Format"


def analyze_quality_flags(df, instrument_name):
    """Analyze quality flags for a specific instrument."""
    quality_column = "QUALITY_mdi" if instrument_name == "SOHO/MDI" else "QUALITY_hmi"

    if quality_column not in df.columns or len(df) == 0:
        return None

    series = (
        df[quality_column]
        .astype(str)
        .replace(["nan", "None", "<NA>", ""], "00000000")
        .str.strip()
        .str.replace("0x", "", regex=False)
    )

    counts = series.value_counts().reset_index()
    counts.columns = ["Flag", "Count"]
    total = counts["Count"].sum()

    return (
        counts.assign(
            Percentage=(counts["Count"] / total * 100).round(2).apply(lambda p: f"{p:.2f}%"),
            Flag_Hex=counts["Flag"].apply(lambda f: f"0x{f.upper()}"),
            Description=counts["Flag"].apply(lambda f: decode_flags(f, QUALITY_FLAGS[instrument_name])),
        )[["Flag_Hex", "Count", "Percentage", "Description"]]
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )


# %%
quality_mdi_df = analyze_quality_flags(df_MDI, "SOHO/MDI")
quality_mdi_df

# %%
quality_hmi_df = analyze_quality_flags(df_HMI, "SDO/HMI")
quality_hmi_df


# %% [markdown]
# ## Data Filtering

# %%
# Remove bad quality data
hmi_good_flags = ["", "0x00000000", "0x00000400"]
mdi_good_flags = ["", "00000000", "00000200"]

df_clean = df[df["QUALITY_hmi"].isin(hmi_good_flags) & df["QUALITY_mdi"].isin(mdi_good_flags)]
df_HMI_clean = df_HMI[df_HMI["QUALITY_hmi"].isin(hmi_good_flags)]
df_MDI_clean = df_MDI[df_MDI["QUALITY_mdi"].isin(mdi_good_flags)]

print("DATA FILTERING Stats")
print("-" * 40)
hmi_orig, hmi_clean = len(df_HMI), len(df_HMI_clean)
mdi_orig, mdi_clean = len(df_MDI), len(df_MDI_clean)
total_orig, total_clean = len(df), len(df_clean)

print(f"HMI: {hmi_clean:,}/{hmi_orig:,} ({hmi_clean / hmi_orig * 100:.1f}% retained)")
print(f"MDI: {mdi_clean:,}/{mdi_orig:,} ({mdi_clean / mdi_orig * 100:.1f}% retained)")
print(f"Total: {total_clean:,}/{total_orig:,} ({total_clean / total_orig * 100:.1f}% retained)")
print("-" * 40)

# %%
# Path Analysis
# Count rows where both paths are empty
both_empty = (df_clean["path_image_cutout_hmi"] == "") & (df_clean["path_image_cutout_mdi"] == "")
both_empty_count = both_empty.sum()

# Count rows where at least one path exists
hmi_exists = df_clean["path_image_cutout_hmi"] != ""
mdi_exists = df_clean["path_image_cutout_mdi"] != ""
at_least_one_exists = (hmi_exists | mdi_exists).sum()

print("PATH ANALYSIS:")
print("-" * 40)
print(f"Total rows in df_clean: {len(df_clean):,}")
print(f"Both paths empty: {both_empty_count:,} ({both_empty_count / len(df_clean) * 100:.1f}%)")
print(f"At least one path exists: {at_least_one_exists:,}")
print(f"Both paths exist: {(hmi_exists & mdi_exists).sum():,}")

# Remove rows where both paths are empty
df_clean = df_clean[~both_empty].copy()

# %%
# Reset index after filtering to ensure continuous indexing
df_clean = df_clean.reset_index(drop=True)

# %% [markdown]
# # Location of ARs on the Sun

# %%
AR_IA_lbs = ["Alpha", "Beta", "IA", "Beta-Gamma-Delta", "Beta-Gamma", "Beta-Delta", "Gamma-Delta", "Gamma"]
AR_IA_df = df_clean[df_clean["label"].isin(AR_IA_lbs)]

# %%
ut_v.make_classes_histogram(AR_IA_df["label"], figsz=(12, 7), text_fontsize=11, title=f"{dataset_title} ARs", y_off=100)
plt.show()


# %%
def get_coordinates(df, coord_type):
    """Extract longitude or latitude coordinates"""
    hmi_col = f"{coord_type}_hmi"
    mdi_col = f"{coord_type}_mdi"
    return np.deg2rad(np.where(df["path_image_cutout_hmi"] != "", df[hmi_col], df[mdi_col]))


def plot_histogram(ax, data, degree_ticks, title, color="#4C72B0"):
    """Plot histogram with degree labels."""
    rad_ticks = np.deg2rad(degree_ticks)
    ax.hist(data, bins=rad_ticks, color=color, edgecolor="black")
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels([f"{deg}°" for deg in degree_ticks])
    ax.set_xlabel(f"{title} (degrees)")
    ax.set_ylabel("Frequency")


# Get coordinates
lonV = get_coordinates(AR_IA_df, "longitude")
latV = get_coordinates(AR_IA_df, "latitude")
degree_ticks = np.arange(-90, 91, 15)

# Plot histograms
with sns.axes_style("darkgrid"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_histogram(ax1, lonV, degree_ticks, "Longitude")
    plot_histogram(ax2, latV, degree_ticks, "Latitude")
    plt.tight_layout()
    plt.show()


# %%
def create_solar_grid(ax, num_meridians=12, num_parallels=12, num_points=300):
    """Add meridian and parallel grid lines to solar disc plot."""
    phis = np.linspace(0, 2 * np.pi, num_meridians, endpoint=False)
    lats = np.linspace(-np.pi / 2, np.pi / 2, num_parallels)
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)

    # Meridians
    for phi in phis:
        y, z = np.cos(theta) * np.sin(phi), np.sin(theta)
        ax.plot(y, z, "k-", linewidth=0.2)

    # Parallels
    for lat in lats:
        y = np.cos(lat) * np.sin(theta)
        z = np.full(num_points, np.sin(lat))
        ax.plot(y, z, "k-", linewidth=0.2)


def filter_by_longitude(df, lonV, latV, long_limit_deg=65):
    """Filter data by longitude and return front/rear coordinates."""
    condition = np.abs(lonV) > np.deg2rad(long_limit_deg)

    # Calculate y, z coordinates
    yV = np.cos(latV) * np.sin(lonV)
    zV = np.sin(latV)

    return {
        "front": {"y": yV[~condition], "z": zV[~condition], "count": np.sum(~condition)},
        "rear": {"y": yV[condition], "z": zV[condition], "count": np.sum(condition)},
        "filtered_df": df[~condition],
        "rear_df": df[condition],
    }


# Filter and plot
results = filter_by_longitude(AR_IA_df, lonV, latV)

# Create solar disc visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none"))
create_solar_grid(ax)

# Plot data points
ax.scatter(results["rear"]["y"], results["rear"]["z"], s=1, alpha=0.2, color=hmi_color, label="Rear")
ax.scatter(results["front"]["y"], results["front"]["z"], s=1, alpha=0.2, color=mdi_color, label="Front")

# Configure plot
ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
ax.axis("off")
ax.legend(fontsize=12)
plt.show()

# Print statistics
front_count, rear_count = results["front"]["count"], results["rear"]["count"]
total_count = front_count + rear_count
print(f"Rear ARs: {rear_count:,}")
print(f"Front ARs: {front_count:,}")
print(f"Percentage of rear ARs: {100 * rear_count / total_count:.2f}%")

ut_v.make_classes_histogram(results["filtered_df"]["label"], title="Front ARs", y_off=10, figsz=(11, 5))
ut_v.make_classes_histogram(results["rear_df"]["label"], title="Rear ARs", y_off=10, figsz=(11, 5))
plt.show()

# %% [markdown]
# # Time Distribution
# %%
mdi_df = AR_IA_df[AR_IA_df["path_image_cutout_mdi"] != ""]
hmi_df = AR_IA_df[AR_IA_df["path_image_cutout_hmi"] != ""]

# Get time series data
mdi_dates, hmi_dates = mdi_df["dates"].values, hmi_df["dates"].values
mdi_counts, hmi_counts = mdi_df["dates"].value_counts().sort_index(), hmi_df["dates"].value_counts().sort_index()

# Setup plot
tick_dates = [datetime(year, 1, 1) for year in range(1996, 2025, 2)]

with plt.style.context("seaborn-v0_8-darkgrid"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})

    # Top panel: Bar chart
    ax1.bar(mdi_counts.index, mdi_counts.values, width=0.8, color=mdi_color, alpha=0.9, label="MDI")
    ax1.bar(hmi_counts.index, hmi_counts.values, width=0.8, color=hmi_color, alpha=0.9, label="HMI")
    ax1.set(ylabel="n° of ARs per day", ylim=[0, 20], yticks=np.arange(0, 20 + 2, 2))
    ax1.tick_params(axis="y", labelsize=14)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom panel: Timeline
    for dates, color, y_range in zip([mdi_dates, hmi_dates], [mdi_color, hmi_color], [(0.2, 0.8), (1.2, 1.8)]):
        ax2.vlines(dates, *y_range, color=color, alpha=0.9, linewidth=0.5)

    ax2.set(ylim=[0, 2], yticks=[])
    ax2.grid(True, linestyle="--", alpha=0.75)

    # X-axis formatting
    ax2.xaxis_date()
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xticks(tick_dates)
    plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=14)
    ax2.set_xlabel("Time", fontsize=16)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# # McIntosh Classification

# %%
AR_df = df[df["magnetic_class"] != ""].copy()

ut_v.make_classes_histogram(
    AR_df["mcintosh_class"],
    horizontal=True,
    figsz=(10, 18),
    y_off=20,
    x_rotation=0,
    ylabel="Number of Active Regions",
    title="McIntosh Class Distribution",
    ylim=5900,
)
plt.show()

# %%

# McIntosh classification components
AR_df = df[df["magnetic_class"] != ""].copy()
for comp in ["Z_component", "p_component", "c_component"]:
    AR_df[comp] = AR_df["mcintosh_class"].str[{"Z_component": 0, "p_component": 1, "c_component": 2}[comp]]

# Plot parameters and histograms
params = [
    ("Z_component", (10, 6), "Z McIntosh Component"),
    ("p_component", (9, 6), "p McIntosh Component"),
    ("c_component", (6, 6), "c McIntosh Component"),
]

for component, figsz, title in params:
    ut_v.make_classes_histogram(AR_df[component], y_off=50, figsz=figsz, title=title)
# %%
mappings = {
    # Merge D, E, F into LG (LargeGroup)
    "Z_component": {"A": "A", "B": "B", "C": "C", "D": "LG", "E": "LG", "F": "LG", "H": "H"},
    # Merge s and h into sym & a and k into asym
    "p_component": {"x": "x", "r": "r", "s": "sym", "h": "sym", "a": "asym", "k": "asym"},
    # Merge i and c into frag
    "c_component": {"x": "x", "o": "o", "i": "frag", "c": "frag"},
}

# Apply mappings and plot
for comp, mapping in mappings.items():
    AR_df[f"{comp}_grouped"] = AR_df[comp].map(mapping)
    ut_v.make_classes_histogram(
        AR_df[f"{comp}_grouped"],
        y_off=50,
        figsz={"Z_component": (8, 6), "p_component": (6, 6), "c_component": (5, 6)}[comp],
        title=f"{comp.split('_')[0].upper()} McIntosh Component (Grouped)",
    )
    plt.show()


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
print(f"\nn° of classes: {len(AR_df['mcintosh_class'].unique())}")
print("\n------ Grouped McIntosh Classes ------")
grouped_classes = list(
    (AR_df["Z_component_grouped"] + AR_df["p_component_grouped"] + AR_df["c_component_grouped"]).unique()
)
group_and_sort_classes(grouped_classes)
print(f"\nn° of classes: {len(grouped_classes)}")
# %% [markdown]
# # Pixel Values


# %%
def load_and_analyze_fits_pair(idx, df_clean, data_folder, dataset_folder):
    """Load magnetogram and continuum FITS files and compute statistics."""
    # Check if index is valid
    if idx >= len(df_clean):
        raise ValueError(f"Index {idx} is out of range. df_clean has {len(df_clean)} rows (0-{len(df_clean) - 1})")

    row = df_clean.iloc[idx]
    path = row["path_image_cutout_hmi"] if row["path_image_cutout_mdi"] == "" else row["path_image_cutout_mdi"]
    fits_magn_filename = os.path.basename(path)
    fits_magn_path = Path(data_folder) / dataset_folder / "data/cutout_classification/fits" / fits_magn_filename
    fits_cont_path = Path(str(fits_magn_path).replace("_mag_", "_cont_"))

    # Check if files exist
    if not fits_magn_path.exists():
        raise FileNotFoundError(f"Magnetogram file not found: {fits_magn_path}")
    if not fits_cont_path.exists():
        raise FileNotFoundError(f"Continuum file not found: {fits_cont_path}")

    # Load data
    with fits.open(fits_magn_path) as hdul:
        mag_data = hdul[0].data
    with fits.open(fits_cont_path) as hdul:
        cont_data = hdul[0].data

    # Check if data is not empty
    if mag_data is None or mag_data.size == 0:
        raise ValueError(f"Magnetogram data is empty: {fits_magn_filename}")
    if cont_data is None or cont_data.size == 0:
        raise ValueError(f"Continuum data is empty: {fits_cont_path.name}")

    # Compute statistics
    def compute_stats(data):
        return {
            "mean": np.nanmean(data),
            "median": np.nanmedian(data),
            "std": np.nanstd(data),
            "min": np.nanmin(data),
            "max": np.nanmax(data),
            "shape": data.shape,
        }

    return {
        "row": row,
        "mag_data": mag_data,
        "cont_data": cont_data,
        "mag_stats": compute_stats(mag_data),
        "cont_stats": compute_stats(cont_data),
        "mag_filename": fits_magn_filename,
        "cont_filename": fits_cont_path.name,
    }


idx = 1195

data = load_and_analyze_fits_pair(idx, df_clean, data_folder, dataset_folder)

# Create subplot and display
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{data['row']['label']} - {data['row']['mcintosh_class']} - {data['row']['dates']}", fontsize=12, y=0.95)

# Display both images with colorbars
for ax, img_data, title, stats in zip(
    [ax1, ax2],
    [data["mag_data"], data["cont_data"]],
    ["Magnetogram", "Continuum"],
    [data["mag_stats"], data["cont_stats"]],
):
    im = ax.imshow(img_data, cmap="gray")
    ax.set_title(f"{title}\nMean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()

# Print statistics
for name, stats in [("MAGNETOGRAM", data["mag_stats"]), ("CONTINUUM", data["cont_stats"])]:
    print(f"{name} STATISTICS:")
    print("-" * 30)
    for key, value in stats.items():
        print(f"{key.capitalize()}: {value}" if key == "shape" else f"{key.capitalize()}: {value:.4f}")
    print()

print(f"Files:\nMagnetogram: {data['mag_filename']}\nContinuum: {data['cont_filename']}")
