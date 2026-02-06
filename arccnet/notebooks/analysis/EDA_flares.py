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

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from arccnet.models.flares import utils as ut_f
from arccnet.notebooks.analysis import EDA_flares_utils as ut_f_eda
from arccnet.visualisation import utils as ut_v

sns.set_style("darkgrid")
pd.set_option("display.max_columns", None)


# %%
# Setup
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
df_flares_name = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"
df_flares = pd.read_parquet(os.path.join(data_folder, df_flares_name))

dataset_folder = "arccnet-v20251017/04_final"
df_file_name = os.path.join(dataset_folder, "cutout-mcintosh-catalog-v20251017.parq")

# %%
ut_v.make_classes_histogram(
    df_flares["magnetic_class"], y_off=50, figsz=(12, 6), fontsize=12, title="ARs Magnetic Classes"
)
plt.show()

# %%
flare_counts = [df_flares[cls].sum() for cls in ut_f_eda.FLARE_CLASSES]
flare_series = pd.Series(np.repeat(ut_f_eda.FLARE_CLASSES, flare_counts))
ut_v.make_classes_histogram(
    flare_series,
    ylabel="n° of Flares",
    y_off=50,
    figsz=(7, 6),
    fontsize=12,
)
plt.show()

# %%
mag_flare_data = ut_f_eda.get_flare_data_by_magnetic_class(df_flares)

with plt.style.context("seaborn-v0_8-darkgrid"):
    plt.figure(figsize=(13, 5))
    ut_f_eda.create_stacked_bar_chart(
        mag_flare_data,
        "magnetic_class",
        ut_f_eda.FLARE_CLASSES,
        colors=dict(zip(ut_f_eda.FLARE_CLASSES, sns.color_palette("Blues", len(ut_f_eda.FLARE_CLASSES)))),
        show_totals=True,
        title="Solar Flare Distribution by Magnetic Class",
        xlabel="Magnetic Class",
        ylabel="Number of Flares",
    )
    plt.legend(title="Flare Class", fontsize=12)
    plt.tight_layout()
    plt.show()

mag_flare_data

# %%
ut_f_eda.analyze_flaring_vs_nonflaring(df_flares)

# %%
df_flares_exists, none_idxs = ut_f.check_fits_file_existence(df_flares.copy(), data_folder, dataset_folder)
plt.figure(figsize=(6, 6))
ax = sns.countplot(x="file_exists", data=df_flares_exists)
plt.title("File Existence")
plt.xlabel("File Exists Locally")
plt.ylabel("Count")
plt.xticks([0, 1], ["False", "True"])

# Add counts and percentages to the bars
total = len(df_flares_exists)
for p in ax.patches:
    count = p.get_height()
    percentage = f"{100 * count / total:.1f}%"
    x = p.get_x() + p.get_width() / 2
    y = count + 200
    ax.annotate(f"{count} ({percentage})", (x, y), ha="center")

plt.show()

# %%
df_flares_data = df_flares_exists[df_flares_exists["file_exists"]].copy()

# %%
flare_classes = ["A", "B", "C", "M", "X"]

for i in range(len(flare_classes)):
    threshold_class = flare_classes[i]
    columns_to_check = flare_classes[i:]  # Select columns from current class onward
    df_flares_data[f"flares_above_{threshold_class}"] = (
        (df_flares_data[columns_to_check].fillna(0) > 0).any(axis=1).astype(int)
    )

# %%
# Define flare columns
flare_columns = ["flares_above_A", "flares_above_B", "flares_above_C", "flares_above_M", "flares_above_X"]

# Calculate flaring/quiet counts and percentages
total_ars = len(df_flares_data)
flare_summary = pd.DataFrame(
    {
        "Threshold": [col.split("_")[-1] for col in flare_columns],
        "Flaring": df_flares_data[flare_columns].sum().values,
        "Quiet": total_ars - df_flares_data[flare_columns].sum().values,
    }
)

# Compute percentages
flare_summary["Flaring_Pct"] = (flare_summary["Flaring"] / total_ars * 100).round(1)
flare_summary["Quiet_Pct"] = (flare_summary["Quiet"] / total_ars * 100).round(1)

# Melt to long format for seaborn
flare_summary_melted = flare_summary.melt(
    id_vars=["Threshold"], value_vars=["Flaring", "Quiet"], var_name="Status", value_name="Count"
)

# Plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x="Threshold",
    y="Count",
    hue="Status",
    data=flare_summary_melted,
    palette={"Flaring": "#ff7f0e", "Quiet": "#1f77b4"},
)

plt.title("Active Regions: Flaring vs. Quiet", fontsize=14)
plt.xlabel("Flare Threshold", fontsize=12)
plt.ylabel("Count of ARs", fontsize=12)
plt.xticks(rotation=0)

# Annotate with counts AND percentages
for i, threshold in enumerate(flare_summary["Threshold"]):
    flaring_count = flare_summary.loc[i, "Flaring"]
    quiet_count = flare_summary.loc[i, "Quiet"]
    flaring_pct = flare_summary.loc[i, "Flaring_Pct"]
    quiet_pct = flare_summary.loc[i, "Quiet_Pct"]

    # Annotate Flaring bars (top)
    ax.text(
        i - 0.2,
        flaring_count + 5,
        f"{flaring_count}\n({flaring_pct}%)",
        ha="center",
        va="bottom",
        color="#ff7f0e",
        fontsize=10,
    )

    # Annotate Quiet bars (top)
    ax.text(
        i + 0.2,
        quiet_count + 5,
        f"{quiet_count}\n({quiet_pct}%)",
        ha="center",
        va="bottom",
        color="#1f77b4",
        fontsize=10,
    )
plt.ylim([0, 36000])
plt.tight_layout()
plt.show()
