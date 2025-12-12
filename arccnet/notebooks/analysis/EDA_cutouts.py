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
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import os
from datetime import datetime
from collections import defaultdict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from p_tqdm import p_map

from astropy.time import Time

from arccnet import load_config
from arccnet.models import dataset_utils as ut_d
from arccnet.visualisation import utils as ut_v
from arccnet.visualisation.EDA_utils import (
    analyze_quality_flags,
    create_solar_grid,
    load_and_analyze_fits_pair,
    process_row,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
config = load_config()

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
dataset_folder = "arccnet-v20251017/04_final"
df_file_name = "data/cutout_classification/region_classification.parq"
dataset_title = "arccnet v20251017"

# %%
# complete dataframe
df_raw = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_file_name))
df_raw["label"] = np.where(df_raw["magnetic_class"] == "", df_raw["region_type"], df_raw["magnetic_class"])
ut_v.make_classes_histogram(df_raw["label"], figsz=(18, 6), text_fontsize=11, title=dataset_title)
plt.show()


# %% [markdown]
# # Quality Flags


# %%
# Filter out rows without either quicklook path
def has_quicklook_path(df: pd.DataFrame) -> pd.Series:
    """Return boolean mask for rows with at least one valid quicklook path."""
    has_mdi = df["quicklook_path_mdi"].notna() & (df["quicklook_path_mdi"] != "") & (df["quicklook_path_mdi"] != "None")
    has_hmi = df["quicklook_path_hmi"].notna() & (df["quicklook_path_hmi"] != "") & (df["quicklook_path_hmi"] != "None")
    return has_mdi | has_hmi


df_with_quicklook = df_raw[has_quicklook_path(df_raw)].copy()
print(
    f"Filtered dataframe: {len(df_with_quicklook):,} rows retained from {len(df_raw):,} ({len(df_with_quicklook) / len(df_raw) * 100:.1f}%)"
)


# %%
def propagate_quality(df: pd.DataFrame, instrument: str = "hmi") -> pd.DataFrame:
    qual_col = f"QUALITY_{instrument}"
    qual_mask_col = f"{qual_col}.mask"
    qlk_col = f"quicklook_path_{instrument}"
    qlk_mask_col = f"{qlk_col}.mask"

    # keep rows with a usable quicklook key
    keyed = df[~df[qlk_mask_col] & df[qlk_col].notna()].copy()

    # canonical quality values (mask/empty/nan -> NA)
    qvals = keyed[qual_col].mask(keyed[qual_mask_col])
    qvals = qvals.astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})

    # consensus quality per quicklook (only if exactly one non-empty unique value)
    consensus = (
        qvals.groupby(keyed[qlk_col])
        .agg(lambda s: s.dropna().unique())
        .loc[lambda s: s.str.len() == 1]
        .map(lambda arr: arr[0])
    )

    # map consensus back to all rows sharing that quicklook
    fill_values = df[qlk_col].map(consensus)
    to_fill = fill_values.notna()
    df.loc[to_fill, qual_col] = fill_values[to_fill]
    df.loc[to_fill, qual_mask_col] = False
    return df


df_quality_propagated = propagate_quality(df_with_quicklook.copy(), "hmi")
df_quality_propagated = propagate_quality(df_quality_propagated, "mdi")

# %%
mdi_color = "royalblue"
hmi_color = "tomato"

df_raw_mdi_only = df_raw[df_raw["path_image_cutout_hmi"] == ""].copy()
df_raw_hmi_only = df_raw[df_raw["path_image_cutout_mdi"] == ""].copy()

df_quality_propagated_mdi_only = df_quality_propagated[df_quality_propagated["path_image_cutout_hmi"] == ""].copy()
df_quality_propagated_hmi_only = df_quality_propagated[df_quality_propagated["path_image_cutout_mdi"] == ""].copy()

# %%
quality_mdi_df = analyze_quality_flags(df_raw_mdi_only, "SOHO/MDI")
quality_mdi_df

# %%
quality_hmi_df = analyze_quality_flags(df_raw_hmi_only, "SDO/HMI")
quality_hmi_df


# %%
# filtered df
analyze_quality_flags(df_quality_propagated_hmi_only, "SDO/HMI")

# %%
analyze_quality_flags(df_quality_propagated_mdi_only, "SOHO/MDI")

# %% [markdown]
# ## Data Filtering

# %%
# Remove bad quality data using df_quality_propagated (with propagated quality flags)
hmi_good_flags = ["0x00000000", "0x00000400"]
mdi_good_flags = ["00000000", "00000200"]

# Filter based on quality flags - rows need good HMI OR good MDI (not both required)
hmi_good = df_quality_propagated["QUALITY_hmi"].isin(hmi_good_flags) | (
    df_quality_propagated["path_image_cutout_hmi"] == ""
)
mdi_good = df_quality_propagated["QUALITY_mdi"].isin(mdi_good_flags) | (
    df_quality_propagated["path_image_cutout_mdi"] == ""
)
df_quality_filtered = df_quality_propagated[hmi_good & mdi_good].copy()

df_quality_filtered_hmi = df_quality_propagated_hmi_only[
    df_quality_propagated_hmi_only["QUALITY_hmi"].isin(hmi_good_flags)
]
df_quality_filtered_mdi = df_quality_propagated_mdi_only[
    df_quality_propagated_mdi_only["QUALITY_mdi"].isin(mdi_good_flags)
]

print("DATA FILTERING Stats")
print("-" * 40)
hmi_orig, hmi_clean_count = len(df_quality_propagated_hmi_only), len(df_quality_filtered_hmi)
mdi_orig, mdi_clean_count = len(df_quality_propagated_mdi_only), len(df_quality_filtered_mdi)
total_orig, total_clean = len(df_quality_propagated), len(df_quality_filtered)

print(f"HMI: {hmi_clean_count:,}/{hmi_orig:,} ({hmi_clean_count / hmi_orig * 100:.1f}% retained)")
print(f"MDI: {mdi_clean_count:,}/{mdi_orig:,} ({mdi_clean_count / mdi_orig * 100:.1f}% retained)")
print(f"Total: {total_clean:,}/{total_orig:,} ({total_clean / total_orig * 100:.1f}% retained)")
print("-" * 40)


# %%
# Path Analysis - Check for None, empty strings, and 'None' strings
def is_missing(series):
    """
    Check if file paths are missing or invalid.
    """
    return series.isna() | (series == "") | (series == "None")


hmi_missing = is_missing(df_quality_filtered["path_image_cutout_hmi"])
mdi_missing = is_missing(df_quality_filtered["path_image_cutout_mdi"])
both_missing = hmi_missing & mdi_missing

# Count statistics
stats = {
    "total": len(df_quality_filtered),
    "hmi_none": (df_quality_filtered["path_image_cutout_hmi"] == "None").sum(),
    "hmi_empty": (df_quality_filtered["path_image_cutout_hmi"] == "").sum(),
    "mdi_none": (df_quality_filtered["path_image_cutout_mdi"] == "None").sum(),
    "mdi_empty": (df_quality_filtered["path_image_cutout_mdi"] == "").sum(),
    "both_missing": both_missing.sum(),
    "at_least_one": (~hmi_missing | ~mdi_missing).sum(),
    "both_exist": (~hmi_missing & ~mdi_missing).sum(),
}

print("PATH ANALYSIS (path_image_cutout columns):")
print("-" * 40)
print(f"Total rows in quality-filtered df: {stats['total']:,}")
print(f"HMI paths - None: {stats['hmi_none']:,}, Empty: {stats['hmi_empty']:,}")
print(f"MDI paths - None: {stats['mdi_none']:,}, Empty: {stats['mdi_empty']:,}")
print(f"Both paths missing: {stats['both_missing']:,} ({stats['both_missing'] / stats['total'] * 100:.1f}%)")
print(f"At least one path exists: {stats['at_least_one']:,}")
print(f"Both paths exist: {stats['both_exist']:,}")

# Remove rows where both paths are missing
df_final = df_quality_filtered[~both_missing].copy()
df_final = df_final.reset_index(drop=True)

print(f"\nAfter removing rows with both paths missing: {len(df_final):,} rows")


# %%
df_quality_filtered.loc[both_missing]


# %%
def _convert_jd_to_datetime(df):
    """Convert Julian dates to datetime objects."""
    df["time"] = df["target_time.jd1"] + df["target_time.jd2"]
    times = Time(df["time"], format="jd")
    df["dates"] = pd.to_datetime(times.iso)
    return df


final_df = _convert_jd_to_datetime(df_final)
final_df

# %%
ut_v.make_classes_histogram(final_df["label"], figsz=(14, 6), text_fontsize=11, title=f"{dataset_title} - Processed")
plt.show()

# %% [markdown]
# # Location of ARs on the Sun

# %%
AR_IA_lbs = ["Alpha", "Beta", "IA", "Beta-Gamma-Delta", "Beta-Gamma", "Beta-Delta", "Gamma-Delta", "Gamma"]
AR_IA_df = final_df[final_df["label"].isin(AR_IA_lbs)].reset_index(drop=True)

# %%
ut_v.make_classes_histogram(AR_IA_df["label"], figsz=(12, 7), text_fontsize=11, title=f"{dataset_title} ARs", y_off=100)
plt.show()


# %%
def get_coordinates(df, coord_type):
    """Extract longitude or latitude coordinates"""
    hmi_col = f"{coord_type}_hmi"
    mdi_col = f"{coord_type}_mdi"
    return np.deg2rad(np.where(df["path_image_cutout_hmi"] != "", df[hmi_col], df[mdi_col]))


def plot_histogram(ax, data, degree_ticks, degree_bins, title, color="#4C72B0"):
    """Plot histogram with degree labels."""
    rad_ticks = np.deg2rad(degree_ticks)
    rad_bins = np.deg2rad(degree_bins)
    ax.hist(data, bins=rad_bins, color=color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(rad_ticks)
    ax.set_xticklabels([f"{deg}°" for deg in degree_ticks])
    ax.set_xlabel(f"{title} (degrees)")
    ax.set_ylabel("Frequency")


# Get coordinates
lonV = get_coordinates(AR_IA_df, "longitude")
latV = get_coordinates(AR_IA_df, "latitude")
degree_ticks = np.arange(-90, 91, 15)
degree_bins = np.arange(-90, 91, 2)

# Plot histograms
with plt.style.context("seaborn-v0_8-darkgrid"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_histogram(ax1, lonV, degree_ticks, degree_bins, title="Longitude")
    plot_histogram(ax2, latV, degree_ticks, degree_bins, title="Latitude")
    plot_histogram(ax1, lonV, degree_ticks, degree_bins, title="Longitude")
    plot_histogram(ax2, latV, degree_ticks, degree_bins, title="Latitude")
    plt.tight_layout()
    plt.show()


# %%
results = ut_d.filter_by_location(AR_IA_df, lon_limit_deg=65)

# Get coordinates for visualization
lonV = get_coordinates(AR_IA_df, "longitude")
latV = get_coordinates(AR_IA_df, "latitude")

# Calculate y, z coordinates for plotting
yV = np.cos(latV) * np.sin(lonV)
zV = np.sin(latV)

# Create solar disc visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_artist(plt.Circle((0, 0), 1, edgecolor="gray", facecolor="none"))
create_solar_grid(ax)

# Plot data points using masks
front_mask, rear_mask = results["mask_front"], results["mask_rear"]
ax.scatter(yV[rear_mask], zV[rear_mask], s=1, alpha=0.2, color=hmi_color, label="Rear")
ax.scatter(yV[front_mask], zV[front_mask], s=1, alpha=0.2, color=mdi_color, label="Front")

# Configure plot
ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal")
ax.axis("off")
ax.legend(fontsize=12)
plt.show()

# Print statistics
front_count = int(front_mask.sum())
rear_count = int(rear_mask.sum())
total_count = front_count + rear_count
print(f"Rear ARs: {rear_count:,}")
print(f"Front ARs: {front_count:,}")
print(f"Percentage of rear ARs: {100 * rear_count / total_count:.2f}%")

ut_v.make_classes_histogram(results["df_front"]["label"], title="Front ARs", y_off=10, figsz=(11, 5))
ut_v.make_classes_histogram(results["df_rear"]["label"], title="Rear ARs", y_off=10, figsz=(11, 5))
plt.show()

# %% [markdown]
# # Quality Data Issues

# %%
problematic_quicklooks = [
    ql.strip().lstrip("\\").strip() for ql in config.get("magnetograms", "problematic_quicklooks").split(",")
]

# %%
final_cleaned_df = AR_IA_df[~AR_IA_df["quicklook_path_hmi"].isin(problematic_quicklooks)].reset_index(drop=True)
basenames = final_df["quicklook_path_mdi"].apply(os.path.basename)
mask = basenames.isin(problematic_quicklooks)
problematic_rows = final_df[mask]
problematic_rows.to_parquet("problematic_rows_good_quality.parq")

# %%
problematic_rows["QUALITY_mdi"].value_counts()

# %%
problematic_rows[problematic_rows["QUALITY_mdi"] == "00000000"]["quicklook_path_mdi"].unique()

# %%
problematic_rows[problematic_rows["QUALITY_mdi"] == "00000200"]["quicklook_path_mdi"].unique()


# %%
def _remove_problematic_quicklooks(df):
    """Remove problematic magnetograms from the dataset."""

    def is_problematic(path):
        """Check if a path's basename matches any problematic quicklook."""
        if not path or pd.isna(path) or path == "" or path == "None":
            return False
        return os.path.basename(path) in problematic_quicklooks

    # Check both MDI and HMI quicklook paths
    mask_mdi = df["quicklook_path_mdi"].apply(is_problematic)
    mask_hmi = df["quicklook_path_hmi"].apply(is_problematic)
    mask = mask_mdi | mask_hmi
    filtered_df = df[mask]
    df = df[~mask].reset_index(drop=True)
    return df, filtered_df


# Apply the function to final_df
final_df_cleaned, problematic_final_df = _remove_problematic_quicklooks(final_df)

print(f"Original dataframe: {len(final_df):,} rows")
print(f"Cleaned dataframe: {len(final_df_cleaned):,} rows")
print(
    f"Problematic rows removed: {len(problematic_final_df):,} rows ({len(problematic_final_df) / len(final_df) * 100:.2f}%)"
)

# %%
# Create AR_IA_df from cleaned dataframe
AR_IA_df = final_df_cleaned[final_df_cleaned["label"].isin(AR_IA_lbs)].reset_index(drop=True)

print(f"AR_IA_df created from cleaned data: {len(AR_IA_df):,} rows")
ut_v.make_classes_histogram(
    AR_IA_df["label"], figsz=(12, 7), text_fontsize=11, title=f"{dataset_title} ARs (Cleaned)", y_off=100
)
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
AR_df = df_with_quicklook[df_with_quicklook["magnetic_class"] != ""].copy()

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
AR_df = df_with_quicklook[df_with_quicklook["magnetic_class"] != ""].copy()
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
    """
    Group classes by their initial letter and display them.
    """
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


# %% [markdown]
# ### Single Image


# %%
idx = 144

data = load_and_analyze_fits_pair(idx, AR_IA_df, data_folder, dataset_folder)

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
    im = ax.imshow(img_data, cmap="gray", origin="lower")
    ax.set_title(f"{title}\nMean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()

print(f"Label: {data['row']['label']} - {data['row']['mcintosh_class']}")
print(f"Date: {data['row']['dates']}")
print(f"{'Statistic':<12} {'Magnetogram':<12} {'Continuum':<12}")
print("-" * 36)
print(f"{'Mean':<12} {data['mag_stats']['mean']:<12.2f} {data['cont_stats']['mean']:<12.2f}")
print(f"{'Std Dev':<12} {data['mag_stats']['std']:<12.2f} {data['cont_stats']['std']:<12.2f}")
print(f"{'Min':<12} {data['mag_stats']['min']:<12.2f} {data['cont_stats']['min']:<12.2f}")
print(f"{'Max':<12} {data['mag_stats']['max']:<12.2f} {data['cont_stats']['max']:<12.2f}")

# %% [markdown]
# ### All images statistics


# %%
def process_row_wrapper(idx):
    """Wrapper function that uses global variables for parallel processing."""
    return process_row(idx, AR_IA_df, data_folder, dataset_folder)


results = p_map(process_row_wrapper, range(len(AR_IA_df)))

# %%
flat_stats = []
for entry in results:
    if entry is not None:
        idx = entry["index"]
        row = {
            "index": idx,
            "label": entry["label"],
            "mcintosh_class": entry["mcintosh_class"],
        }
        row.update({f"mag_{k}": v for k, v in entry["mag_stats"].items()})
        row.update({f"cont_{k}": v for k, v in entry["cont_stats"].items()})
        flat_stats.append(row)
stats_df = pd.DataFrame(flat_stats)
stats_df.describe()

# %%
# Find the indices of the 10 highest mag_mean values
top10_indices = stats_df["mag_mean"].nlargest(10).index
# Get the corresponding rows
top10_rows = stats_df.loc[top10_indices]
top10_rows

# %%
stats_config = [
    ("mag_mean", "Magnetogram Mean", "royalblue"),
    ("mag_std", "Magnetogram Std Dev", "royalblue"),
    ("mag_min", "Magnetogram Min", "royalblue"),
    ("mag_max", "Magnetogram Max", "royalblue"),
    ("cont_mean", "Continuum Mean", "tomato"),
    ("cont_std", "Continuum Std Dev", "tomato"),
    ("cont_min", "Continuum Min", "tomato"),
    ("cont_max", "Continuum Max", "tomato"),
]

# Create histograms
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, (col, title, color) in enumerate(stats_config):
    ax = axes.flat[i]
    ax.hist(stats_df[col], bins=70, color=color, alpha=0.7, edgecolor="black", linewidth=0.5, log=True)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Value", fontsize=14)
    if i % 4 == 0:  # First column gets y-label
        ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()

# %%
# Create boxplots by class
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
all_labels = stats_df["label"].unique()

# Create a custom color palette
palette = {label: colors[i % len(colors)] for i, label in enumerate(all_labels)}

for col, title, _ in stats_config:
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=stats_df, x="label", y=col, hue="label", palette=palette, ax=ax, legend=False)
    ax.set_title(f"{title} by Active Region Class", fontsize=18)
    ax.tick_params(axis="x", rotation=45, labelsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Value", fontsize=16)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(ha="right")
    plt.tight_layout()
    plt.show()
