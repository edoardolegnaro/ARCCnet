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
import logging
from pathlib import Path

import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.data_preparation import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
# Prepare dataset
PROCESSED_DATA_PATH = Path(config.DATA_FOLDER) / "processed_dataset_with_folds.parquet"

if not PROCESSED_DATA_PATH.exists():
    logging.info("Preparing dataset for the first time...")
    df_processed = prepare_dataset(save_path=str(PROCESSED_DATA_PATH))
else:
    logging.info("Loading previously prepared dataset...")
    df_processed = pd.read_parquet(str(PROCESSED_DATA_PATH))
