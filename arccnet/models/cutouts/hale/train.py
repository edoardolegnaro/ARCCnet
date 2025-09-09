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

import arccnet.models.cutouts.hale.config as config
from arccnet.models import dataset_utils as ut_d

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
# Load and process dataset in streamlined pipeline
df, AR_df, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
logging.info(f"Original DataFrame shape: {df.shape}")

# Clean data
df_clean = ut_d.cleanup_df(df)
logging.info(f"After cleanup: {df_clean.shape} ({len(df_clean) / len(df) * 100:.1f}% retained)")

# %%
df_original, df_processed = ut_d.undersample_group_filter(
    df_clean, label_mapping=config.label_mapping, long_limit_deg=65, undersample=False
)

logging.info(f"Label mapping applied: {len(df_original):,} → {len(df_processed):,}")
class_dist = df_processed["grouped_labels"].value_counts()
class_pct = df_processed["grouped_labels"].value_counts(normalize=True) * 100
logging.info("Final class distribution:")
for label in class_dist.index:
    logging.info(f"  {label}: {class_dist[label]:,} ({class_pct[label]:.1f}%)")

# %%
