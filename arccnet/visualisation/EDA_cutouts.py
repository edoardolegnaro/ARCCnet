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
config = load_config()


# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
dataset_folder = "arccnet-v20250805/04_final"
df_file_name = "data/cutout_classification/region_classification.parq"

# %%
df, _ = ut_d.make_dataframe(data_folder, dataset_folder, df_file_name)
ut_v.make_classes_histogram(df["label"], figsz=(18, 6), text_fontsize=11, title="arccnet v20250805")
