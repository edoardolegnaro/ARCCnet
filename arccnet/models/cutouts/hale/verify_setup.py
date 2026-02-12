#!/usr/bin/env python3
"""Quick verification that everything is set up correctly."""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    # Test 1: Import config
    logger.info("Test 1: Loading configuration...")
    import arccnet.models.cutouts.hale.config as config

    logger.info("✓ Config loaded")
    logger.info(f"  - Data folder: {config.DATA_FOLDER}")
    logger.info(f"  - Dataset folder: {config.DATASET_FOLDER}")
    logger.info(f"  - DF file: {config.DF_FILE_NAME}")

    # Test 2: Check parquet file exists
    logger.info("Test 2: Checking parquet file...")
    full_path = Path(config.DATA_FOLDER) / config.DATASET_FOLDER / config.DF_FILE_NAME
    if full_path.exists():
        logger.info(f"✓ File exists: {full_path}")
        logger.info(f"  Size: {full_path.stat().st_size / 1e6:.1f}MB")
    else:
        logger.error(f"✗ File NOT found: {full_path}")
        sys.exit(1)

    # Test 3: Load data
    logger.info("Test 3: Loading dataset...")
    from arccnet.models.cutouts.hale.data_preparation import load_and_clean_dataset

    df, df_clean = load_and_clean_dataset(nan_threshold=0.05)
    logger.info("✓ Dataset loaded successfully")
    logger.info(f"  - Raw shape: {df.shape}")
    logger.info(f"  - Clean shape: {df_clean.shape}")
    logger.info(f"  - Path columns: {[c for c in df_clean.columns if 'path' in c.lower()][:3]}")

    logger.info("\n✓ All checks passed! Ready to train.")

except Exception as e:
    logger.error(f"\n✗ Error: {e}", exc_info=True)
    sys.exit(1)
