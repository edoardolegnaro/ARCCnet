# -*- coding: utf-8 -*-
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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from arccnet.visualisation import utils as ut_v
from arccnet.visualisation import EDA_flares_utils as ut_f


# %%
# Setup
pd.set_option("display.max_columns", None)
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../data")
df_file_name = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"
df = pd.read_parquet(os.path.join(data_folder, df_file_name))

# %%
ut_v.make_classes_histogram(
    df['magnetic_class'], 
    y_off=50, 
    figsz=(12,6), 
    fontsize=12, 
    title='Magnetic Classes')
plt.show()

# %%
flare_counts = [df[cls].sum() for cls in ut_f.FLARE_CLASSES]
flare_series = pd.Series(np.repeat(ut_f.FLARE_CLASSES, flare_counts))
ut_v.make_classes_histogram(
    flare_series, 
    ylabel="n° of Flares",
    y_off = 50,
    figsz = (7,6),
    fontsize=12, 
    )
plt.show()

# %%
mag_flare_data = ut_f.get_flare_data_by_magnetic_class(df)
mag_flare_data

# %%
with plt.style.context("seaborn-v0_8-darkgrid"):
    plt.figure(figsize=(13, 5))
    ut_f.create_stacked_bar_chart(
        mag_flare_data, 'magnetic_class', ut_f.FLARE_CLASSES, 
        colors=dict(zip(ut_f.FLARE_CLASSES, sns.color_palette("Blues", len(ut_f.FLARE_CLASSES)))),
        show_totals=True,
        title='Solar Flare Distribution by Magnetic Class',
        xlabel='Magnetic Class', 
        ylabel='Number of Flares'
    )
    plt.legend(title='Flare Class', fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
ut_f.analyze_flaring_vs_nonflaring(df)

# %%
