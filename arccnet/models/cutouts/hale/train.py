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

import logging

# %%
import arccnet.models.cutouts.hale.config as config
from arccnet.models import dataset_utils as ut_d

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
df, _, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
logging.info(f"DataFrame shape: {df.shape}")
df_clean = ut_d.cleanup_df(df)
logging.info(f"Cleaned DataFrame shape: {df_clean.shape}")
