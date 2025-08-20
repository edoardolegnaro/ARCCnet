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

import pandas as pd

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

# %%
df, _ = ut_d.make_dataframe(data_folder, dataset_folder, df_file_name)
ut_v.make_classes_histogram(df["label"], figsz=(18, 6), text_fontsize=11, title="arccnet v20250805")

# %%
df

# %%
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
