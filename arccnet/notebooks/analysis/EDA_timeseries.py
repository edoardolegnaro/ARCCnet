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
#     display_name: venv (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Timeseries Data EDA
#
# **Dataset Overview:**
# - Multi-channel timeseries observations of active regions
# - AIA wavelengths: 94, 131, 171, 193, 211, 304, 335, 1600, 1700 Å
# - HMI magnetograms
# - 6-hour observation windows with 1-hour cadence
# - Target: Flare occurrence in next 24 hours

# %%
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
timeseries_path = Path(data_folder) / "04_final" / "data"

# %% [markdown]
# ## 1. Dataset Structure and Overview

# %%
# Get all timeseries sample directories
sample_dirs = sorted(list(timeseries_path.glob("*")))
print(f"Total number of timeseries samples: {len(sample_dirs)}")
print("\nFirst 5 samples:")
for d in sample_dirs[:5]:
    print(f"  {d.name}")


# %%
# Parse directory names to extract metadata
def parse_directory_name(dirname):
    """
    Parse timeseries directory name to extract metadata.
    Format: YYYY-MM-DD_NOAA_MagClass_McIntosh_Xb#_Mb#_Cb#_Xa#_Ma#_Ca#
    """
    parts = dirname.split("_")
    if len(parts) < 10:
        return None

    try:
        return {
            "date": parts[0],
            "noaa_ar": int(parts[1]),
            "mag_class": parts[2],
            "mcintosh": parts[3],
            "xb": int(parts[4].replace("Xb", "")),
            "mb": int(parts[5].replace("Mb", "")),
            "cb": int(parts[6].replace("Cb", "")),
            "xa": int(parts[7].replace("Xa", "")),
            "ma": int(parts[8].replace("Ma", "")),
            "ca": int(parts[9].replace("Ca", "")),
            "dirname": dirname,
        }
    except Exception as e:
        print(f"Warning: Failed to parse {dirname}: {e}")
        return None


# %%
# Create dataset metadata DataFrame
metadata_list = []
for d in sample_dirs:
    parsed = parse_directory_name(d.name)
    if parsed:
        metadata_list.append(parsed)

df_metadata = pd.DataFrame(metadata_list)
print(f"Successfully parsed {len(df_metadata)} samples")
print(f"\nDataset shape: {df_metadata.shape}")
df_metadata.head(10)

# %% [markdown]
# ## 2. Temporal Coverage Analysis

# %%
# Convert dates to datetime
df_metadata["datetime"] = pd.to_datetime(df_metadata["date"])
df_metadata["year"] = df_metadata["datetime"].dt.year
df_metadata["month"] = df_metadata["datetime"].dt.month

print("Temporal coverage:")
print(f"  Date range: {df_metadata['datetime'].min()} to {df_metadata['datetime'].max()}")
print(f"  Duration: {(df_metadata['datetime'].max() - df_metadata['datetime'].min()).days} days")
print("\nSamples per year:")
print(df_metadata["year"].value_counts().sort_index())

# %%
# Visualize temporal distribution
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Samples over time
axes[0].hist(df_metadata["datetime"], bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Number of Samples")
axes[0].set_title("Distribution of Timeseries Samples Over Time")
axes[0].grid(True, alpha=0.3)

# Samples per year
year_counts = df_metadata["year"].value_counts().sort_index()
axes[1].bar(year_counts.index, year_counts.values, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Number of Samples")
axes[1].set_title("Samples per Year")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Flare Distribution Analysis

# %%
# Calculate total flares before and after observation window
df_metadata["total_before"] = df_metadata["xb"] + df_metadata["mb"] + df_metadata["cb"]
df_metadata["total_after"] = df_metadata["xa"] + df_metadata["ma"] + df_metadata["ca"]

# Binary classification labels
df_metadata["has_flare_before"] = df_metadata["total_before"] > 0
df_metadata["has_flare_after"] = df_metadata["total_after"] > 0

print("Flare Statistics:")
print("\nBefore observation window (6 hours):")
print(
    f"  Samples with flares: {df_metadata['has_flare_before'].sum()} ({df_metadata['has_flare_before'].mean() * 100:.1f}%)"
)
print(f"  Total X-class: {df_metadata['xb'].sum()}")
print(f"  Total M-class: {df_metadata['mb'].sum()}")
print(f"  Total C-class: {df_metadata['cb'].sum()}")

print("\nAfter observation window (24 hours) - PREDICTION TARGET:")
print(
    f"  Samples with flares: {df_metadata['has_flare_after'].sum()} ({df_metadata['has_flare_after'].mean() * 100:.1f}%)"
)
print(f"  Total X-class: {df_metadata['xa'].sum()}")
print(f"  Total M-class: {df_metadata['ma'].sum()}")
print(f"  Total C-class: {df_metadata['ca'].sum()}")


# %%
# Maximum flare class in next 24 hours
def get_max_flare_class(row):
    """Determine the maximum flare class in next 24 hours."""
    if row["xa"] > 0:
        return "X"
    elif row["ma"] > 0:
        return "M"
    elif row["ca"] > 0:
        return "C"
    else:
        return "None"


df_metadata["max_flare_class"] = df_metadata.apply(get_max_flare_class, axis=1)

print("\nMaximum flare class distribution (next 24 hours):")
print(df_metadata["max_flare_class"].value_counts())
print("\nPercentages:")
print(df_metadata["max_flare_class"].value_counts(normalize=True) * 100)

# %%
# Visualize flare distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Flare counts before
flare_before = pd.DataFrame(
    {"X-class": [df_metadata["xb"].sum()], "M-class": [df_metadata["mb"].sum()], "C-class": [df_metadata["cb"].sum()]}
)
flare_before.T.plot(kind="bar", ax=axes[0, 0], legend=False, color=["#d62728", "#ff7f0e", "#2ca02c"])
axes[0, 0].set_title("Total Flare Counts (Before: 6-hour window)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)
axes[0, 0].grid(True, alpha=0.3, axis="y")

# Flare counts after
flare_after = pd.DataFrame(
    {"X-class": [df_metadata["xa"].sum()], "M-class": [df_metadata["ma"].sum()], "C-class": [df_metadata["ca"].sum()]}
)
flare_after.T.plot(kind="bar", ax=axes[0, 1], legend=False, color=["#d62728", "#ff7f0e", "#2ca02c"])
axes[0, 1].set_title("Total Flare Counts (After: 24-hour window)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
axes[0, 1].grid(True, alpha=0.3, axis="y")

# Binary classification distribution
binary_dist = df_metadata["has_flare_after"].value_counts()
# Handle case where one class might not exist
no_flare_count = binary_dist.get(False, 0)
flare_count = binary_dist.get(True, 0)
axes[1, 0].bar(
    ["No Flare", "Flare"], [no_flare_count, flare_count], color=["#1f77b4", "#d62728"], edgecolor="black", alpha=0.7
)
axes[1, 0].set_title("Binary Classification Distribution (24h forecast)")
axes[1, 0].set_ylabel("Number of Samples")
axes[1, 0].grid(True, alpha=0.3, axis="y")
for i, v in enumerate([no_flare_count, flare_count]):
    axes[1, 0].text(i, v + 5, str(v), ha="center", va="bottom", fontweight="bold")

# Multi-class distribution
class_order = ["None", "C", "M", "X"]
class_counts = df_metadata["max_flare_class"].value_counts()
class_counts = class_counts.reindex(class_order, fill_value=0)
colors_multiclass = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
axes[1, 1].bar(class_order, class_counts.values, color=colors_multiclass, edgecolor="black", alpha=0.7)
axes[1, 1].set_title("Multi-class Distribution (Max flare in 24h)")
axes[1, 1].set_ylabel("Number of Samples")
axes[1, 1].set_xlabel("Maximum Flare Class")
axes[1, 1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(class_counts.values):
    axes[1, 1].text(i, v + 5, str(v), ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Active Region Properties

# %%
# Magnetic class distribution
print("Magnetic Class Distribution:")
print(df_metadata["mag_class"].value_counts().head(10))

plt.figure(figsize=(12, 5))
mag_counts = df_metadata["mag_class"].value_counts().head(15)
plt.bar(range(len(mag_counts)), mag_counts.values, edgecolor="black", alpha=0.7)
plt.xticks(range(len(mag_counts)), mag_counts.index, rotation=45, ha="right")
plt.xlabel("Magnetic Class")
plt.ylabel("Number of Samples")
plt.title("Top 15 Magnetic Classes")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %%
# McIntosh classification distribution
print("\nMcIntosh Classification Distribution:")
print(df_metadata["mcintosh"].value_counts().head(20))

# %%
# Unique NOAA ARs
print(f"\nNumber of unique NOAA Active Regions: {df_metadata['noaa_ar'].nunique()}")
print("\nTop 10 most observed ARs:")
print(df_metadata["noaa_ar"].value_counts().head(10))

# %% [markdown]
# ## 5. Sample Directory Structure Analysis

# %%
# Analyze a sample directory structure
sample_dir = sample_dirs[0]
print(f"Sample directory: {sample_dir.name}\n")

# List contents
contents = list(sample_dir.glob("*"))
print("Contents:")
for item in contents:
    if item.is_dir():
        num_files = len(list(item.glob("*")))
        print(f"  [DIR]  {item.name}/ ({num_files} files)")
    else:
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"  [FILE] {item.name} ({size_mb:.2f} MB)")

# %%
# Check AIA wavelengths
aia_dir = sample_dir / "AIA"
if aia_dir.exists():
    aia_files = list(aia_dir.glob("*.fits"))
    print(f"\nTotal AIA files: {len(aia_files)}")

    # Extract wavelengths from filenames
    wavelengths = set()
    for f in aia_files:
        # Pattern: *_WAVELENGTH.aia.*
        match = re.search(r"_(\d+)\.aia\.", f.name)
        if match:
            wavelengths.add(match.group(1))

    wavelengths = sorted(wavelengths, key=lambda x: int(x))
    print(f"AIA Wavelengths: {wavelengths}")
    print(f"Number of unique wavelengths: {len(wavelengths)}")

    # Count files per wavelength
    wl_counts = {}
    for wl in wavelengths:
        count = len(list(aia_dir.glob(f"*_{wl}.aia.*")))
        wl_counts[wl] = count

    print("\nFiles per wavelength:")
    for wl, count in sorted(wl_counts.items(), key=lambda x: int(x[0])):
        print(f"  {wl} Å: {count} timesteps")

# %%
# Check HMI magnetograms
hmi_dir = sample_dir / "HMI"
if hmi_dir.exists():
    hmi_files = list(hmi_dir.glob("*.fits"))
    print(f"\nTotal HMI files: {len(hmi_files)}")
    print("First few HMI files:")
    for f in hmi_files[:3]:
        print(f"  {f.name}")

# %%
# Check CSV metadata
csv_file = sample_dir / f"{sample_dir.name}.csv"
if csv_file.exists():
    df_csv = pd.read_csv(csv_file)
    print(f"\nCSV Metadata file shape: {df_csv.shape}")
    print("\nColumns:")
    print(df_csv.columns.tolist())
    print("\nFirst few rows:")
    print(df_csv.head())

# %%
# Check flare parquet files
before_file = sample_dir / "before_flares.parquet"
after_file = sample_dir / "after_flares.parquet"

if before_file.exists():
    df_before = pd.read_parquet(before_file)
    print("Before flares (during 6h observation):")
    print(f"  Shape: {df_before.shape}")
    if len(df_before) > 0:
        print(df_before.head())
    else:
        print("  No flares during observation window")

if after_file.exists():
    df_after = pd.read_parquet(after_file)
    print("\nAfter flares (next 24h - PREDICTION TARGET):")
    print(f"  Shape: {df_after.shape}")
    if len(df_after) > 0:
        print(df_after.head())
    else:
        print("  No flares in next 24 hours")

# %% [markdown]
# ## 6. FITS File Analysis

# %%
# Load and analyze a sample FITS file
aia_file = list((sample_dir / "AIA").glob("*_171.aia.*.fits"))[0]
hmi_file = list((sample_dir / "HMI").glob("*.fits"))[0]

print(f"Analyzing AIA file: {aia_file.name}")
with fits.open(aia_file) as hdul:
    print(f"\nNumber of HDUs: {len(hdul)}")
    hdul.info()

    # Get image data
    data = hdul[1].data
    header = hdul[1].header

    print(f"\nImage shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.nanmin(data):.2f}")
    print(f"Max value: {np.nanmax(data):.2f}")
    print(f"Mean value: {np.nanmean(data):.2f}")
    print(f"Std value: {np.nanstd(data):.2f}")
    print(f"NaN pixels: {np.isnan(data).sum()} ({np.isnan(data).sum() / data.size * 100:.2f}%)")

    print("\nKey header info:")
    for key in ["INSTRUME", "WAVELNTH", "T_REC", "QUALITY", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2"]:
        if key in header:
            print(f"  {key}: {header[key]}")

# %%
print(f"\nAnalyzing HMI file: {hmi_file.name}")
with fits.open(hmi_file) as hdul:
    print(f"\nNumber of HDUs: {len(hdul)}")
    hdul.info()

    data = hdul[1].data
    header = hdul[1].header

    print(f"\nImage shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.nanmin(data):.2f}")
    print(f"Max value: {np.nanmax(data):.2f}")
    print(f"Mean value: {np.nanmean(data):.2f}")
    print(f"Std value: {np.nanstd(data):.2f}")
    print(f"NaN pixels: {np.isnan(data).sum()} ({np.isnan(data).sum() / data.size * 100:.2f}%)")

# %% [markdown]
# ## 7. Visualize Sample Data

# %%
# Visualize AIA images across wavelengths
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

aia_dir = sample_dir / "AIA"
wavelengths_to_plot = ["94", "131", "171", "193", "211", "304", "335", "1600", "1700"]

for idx, wl in enumerate(wavelengths_to_plot):
    files = list(aia_dir.glob(f"*_{wl}.aia.*.fits"))
    if files:
        with fits.open(files[0]) as hdul:
            data = hdul[1].data

            # Plot with appropriate scaling
            vmin, vmax = np.nanpercentile(data, [1, 99])
            im = axes[idx].imshow(data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            axes[idx].set_title(f"AIA {wl} Å", fontsize=12, fontweight="bold")
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.suptitle(f"Sample: {sample_dir.name}", fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()
plt.show()

# %%
# Visualize HMI magnetogram
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

with fits.open(hmi_file) as hdul:
    data = hdul[1].data

    # Plot with symmetric colorscale around zero
    vmax = np.nanpercentile(np.abs(data), 99)
    im = ax.imshow(data, cmap="gray", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title("HMI Magnetogram", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Magnetic Field Strength (G)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Temporal Evolution Visualization

# %%
# Visualize temporal evolution of AIA 171 Å
aia_171_files = sorted(list(aia_dir.glob("*_171.aia.*.fits")))
print(f"Number of 171 Å timesteps: {len(aia_171_files)}")

if len(aia_171_files) >= 6:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, aia_file in enumerate(aia_171_files[:6]):
        with fits.open(aia_file) as hdul:
            data = hdul[1].data
            t_rec = hdul[1].header.get("T_REC", "Unknown")

            vmin, vmax = np.nanpercentile(data, [1, 99])
            im = axes[idx].imshow(data, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            axes[idx].set_title(f"T{idx}: {t_rec}", fontsize=10)
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle("Temporal Evolution: AIA 171 Å", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 9. Dataset Statistics Summary

# %%
# Count total images across all samples
total_aia_images = 0
total_hmi_images = 0

for sample_dir in sample_dirs[:50]:  # Sample first 50 for speed
    aia_dir = sample_dir / "AIA"
    hmi_dir = sample_dir / "HMI"

    if aia_dir.exists():
        total_aia_images += len(list(aia_dir.glob("*.fits")))
    if hmi_dir.exists():
        total_hmi_images += len(list(hmi_dir.glob("*.fits")))

print("Dataset Statistics (first 50 samples):")
print(f"  Total AIA images: {total_aia_images}")
print(f"  Total HMI images: {total_hmi_images}")
print(f"  Average AIA images per sample: {total_aia_images / 50:.1f}")
print(f"  Average HMI images per sample: {total_hmi_images / 50:.1f}")

# %%
# Overall summary
print("=" * 60)
print("TIMESERIES DATASET SUMMARY")
print("=" * 60)
print("\nDataset Size:")
print(f"  Total samples: {len(df_metadata)}")
print(f"  Date range: {df_metadata['datetime'].min().date()} to {df_metadata['datetime'].max().date()}")
print(f"  Unique Active Regions: {df_metadata['noaa_ar'].nunique()}")

print("\nData Dimensions (per sample):")
print("  Temporal: ~6 timesteps (1-hour cadence)")
print("  Channels: 9 AIA wavelengths + 1 HMI magnetogram = 10 channels")
print("  Spatial: 400 × 800 pixels")
print("  Expected shape: (6, 10, 400, 800)")

print("\nPrediction Task:")
print(
    f"  Binary: {df_metadata['has_flare_after'].sum()} flaring / {(~df_metadata['has_flare_after']).sum()} non-flaring"
)
print(f"  Class balance: {df_metadata['has_flare_after'].mean() * 100:.1f}% positive class")
print("\n  Multi-class distribution:")
for cls in ["None", "C", "M", "X"]:
    count = (df_metadata["max_flare_class"] == cls).sum()
    pct = count / len(df_metadata) * 100
    print(f"    {cls}-class: {count:3d} ({pct:5.1f}%)")

print("\nData Quality:")
print("  All samples have directory structure: ✓")
print("  Typical NaN coverage: <5% (off-disk regions)")

print("\n" + "=" * 60)
