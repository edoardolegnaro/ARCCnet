import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from astropy.table import Table

from arccnet.models import labels
from arccnet.models.fulldisk import utils as fd_utils
from arccnet.visualisation import utils as ut_v

sns.set_style("darkgrid")
pd.set_option("display.max_columns", None)
# %%
data_folder = Path(os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/"))
dataset_root = Path("arccnet-v20251017")
dataset_folder = dataset_root / "04_final"
df_name = "data/region_detection/region_detection_noaa-xarp.parq"
dataset_title = "arccnet v20251017"

# Load the full dataframe with time columns
tab = Table.read(data_folder / dataset_folder / df_name)
df = fd_utils.load_fulldisk_dataframe(data_folder, dataset_root, dataset_folder, df_name)


# Filter for existing files
def check_file_exists(path):
    """Check if file exists after path conversion."""
    if pd.isna(path):
        return False
    local_path = path.replace("/mnt/ARCAFF/v0.3.0/", str(data_folder / dataset_root) + "/")
    return Path(local_path).exists()


initial_count = len(df)
df["mag_exists"] = df["processed_path_image_mag"].apply(check_file_exists)
df["cont_exists"] = df["processed_path_image_cont"].apply(check_file_exists)
df = df[df["mag_exists"] & df["cont_exists"]].copy()
print(f"File existence check: kept {len(df)}/{initial_count} rows ({len(df) / initial_count * 100:.1f}%)")

# %%
discarded_df = df[df["filtered"]]
selected_df = df[~df["filtered"]]
# Calculate the percentage of selected items and unique fulldisks
selected_percentage = len(selected_df) / len(df) * 100
unique_fulldisks_count = len(selected_df["processed_path_image_mag"].unique())
unique_fulldisks_total = len(df["processed_path_image_mag"].unique())
unique_fulldisks_percentage = unique_fulldisks_count / unique_fulldisks_total * 100
# Print the formatted output
print(f"             Selected FDs: {len(selected_df)} out of {len(df)} ({selected_percentage:.2f}%)")
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
    title="FullDisk Dataset AR classes",
    # ylim=3400,
    transparent=True,
)
plt.show()
# %%
idx = 4567
row = selected_df.iloc[idx]
prefix = "/mnt/ARCAFF/v0.3.0"
local_root = str(data_folder / dataset_root)
image_labels = selected_df[selected_df["processed_path_image_mag"] == row["processed_path_image_mag"]]
fits_magn_path = row["processed_path_image_mag"].replace(prefix, local_root)
fits_cont_path = row["processed_path_image_cont"].replace(prefix, local_root)
fig, (ax_mag, ax_cont), _ = fd_utils.plot_full_disk_pair(
    fits_magnetogram=fits_magn_path,
    fits_continuum=fits_cont_path,
    image_labels=image_labels,
)
fd_utils.add_class_legend(
    fig,
    fd_utils.CLASS_COLOR_MAP,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=min(len(fd_utils.CLASS_COLOR_MAP), 5),
    frameon=False,
)
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Cutout Sizes
fd_utils.w_h_scatterplot(selected_df)
# %%
widths, heights, _, _ = fd_utils.compute_widths_heights(selected_df)
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
# %%
lon_trshld = 70
front_df = fd_utils.filter_front_side(selected_df, longitude_threshold=lon_trshld)
back_df = selected_df[(selected_df["longitude"] >= lon_trshld) | (selected_df["longitude"] <= -lon_trshld)]
# %%
fig, ax = plt.subplots(figsize=(8, 8))
fd_utils.plot_ar_locations(front_df, ax=ax, color="#1f77b4", label="Front Side", marker_size=1, alpha=0.2)
fd_utils.plot_ar_locations(back_df, ax=ax, color="darkorange", label="Back Side", marker_size=1, alpha=0.2)
ax.set_title("ARs Location on the Sun")
ax.legend(loc="upper right", frameon=False)
plt.show()
# %%
fd_utils.w_h_scatterplot(front_df)
# %%
min_size = 0.03
img_size_dic = fd_utils.IMG_SIZE_BY_INSTRUMENT

# Calculate region sizes and filter by minimum size
cleaned_df = fd_utils.calculate_region_sizes(front_df, img_size_dic)
cleaned_df = fd_utils.filter_by_minimum_size(cleaned_df, min_size)
# %%
fd_utils.w_h_scatterplot(cleaned_df)
