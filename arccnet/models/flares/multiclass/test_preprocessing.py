#!/usr/bin/env python3
"""
Test script to verify the on-the-fly data preprocessing works correctly.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from arccnet.models.flares.multiclass import config
from arccnet.models.flares.multiclass.train import (
    filter_solar_limb,
    get_strongest_flare_class,
    preprocess_flare_data,
)
from arccnet.utils.logging import get_logger

logger = get_logger(__name__)


def test_preprocessing():
    """Test the preprocessing pipeline."""
    logger.info("Testing on-the-fly data preprocessing...")

    try:
        # Test the preprocessing function
        df = preprocess_flare_data()
        logger.info(f"✅ Successfully preprocessed {len(df)} records")

        # Check if we have longitude columns for limb filtering
        longitude_cols = [col for col in df.columns if "longitude" in col.lower()]
        logger.info(f"Available longitude columns: {longitude_cols}")

        # Test limb filtering
        df = filter_solar_limb(df)
        logger.info(f"✅ Applied limb filtering, {len(df)} records remaining")

        # Verify the classes are correct
        unique_classes = df[config.TARGET_COLUMN].unique()
        expected_classes = set(config.CLASSES)
        actual_classes = set(unique_classes)

        if actual_classes == expected_classes:
            logger.info(f"✅ Class verification passed: {sorted(actual_classes)}")
        else:
            logger.warning(f"⚠️  Class mismatch. Expected: {expected_classes}, Got: {actual_classes}")

        # Show class distribution
        class_counts = df[config.TARGET_COLUMN].value_counts()
        logger.info("Class distribution:")
        for cls in config.CLASSES:
            count = class_counts.get(cls, 0)
            percentage = (count / len(df)) * 100
            logger.info(f"  {cls}: {count:,} ({percentage:.1f}%)")

        # Check for required columns
        required_cols = ["path_image_cutout_hmi", "path_image_cutout_mdi"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️  Missing required columns: {missing_cols}")
        else:
            logger.info("✅ All required columns present")

        # Test instrument selection logic
        hmi_count = df["path_image_cutout_hmi"].notna().sum()
        mdi_count = df["path_image_cutout_mdi"].notna().sum()
        logger.info(f"HMI observations: {hmi_count}, MDI observations: {mdi_count}")

        logger.info("✅ Preprocessing test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {str(e)}")
        return False


def test_classification_function():
    """Test the classification function with sample data."""
    logger.info("Testing classification function...")

    # Test cases
    test_cases = [
        {"A": 0, "B": 0, "C": 0, "M": 0, "X": 1, "expected": "M_X"},
        {"A": 0, "B": 0, "C": 0, "M": 1, "X": 0, "expected": "M_X"},
        {"A": 0, "B": 0, "C": 0, "M": 1, "X": 1, "expected": "M_X"},  # Both M and X present
        {"A": 0, "B": 0, "C": 1, "M": 0, "X": 0, "expected": "C"},
        {"A": 0, "B": 0, "C": 0, "M": 0, "X": 0, "expected": "Quiet"},
        {"A": 1, "B": 0, "C": 0, "M": 0, "X": 0, "expected": "Filter_out"},
        {"A": 0, "B": 1, "C": 0, "M": 0, "X": 0, "expected": "Filter_out"},
        {"A": 1, "B": 0, "C": 1, "M": 0, "X": 0, "expected": "Filter_out"},  # A class should be filtered even with C
        {"A": 0, "B": 0, "C": 1, "M": 1, "X": 0, "expected": "M_X"},  # M takes precedence over C
    ]

    all_passed = True
    for i, test_case in enumerate(test_cases):
        expected = test_case.pop("expected")
        result = get_strongest_flare_class(test_case)

        if result == expected:
            logger.info(f"✅ Test {i + 1}: {test_case} → {result}")
        else:
            logger.error(f"❌ Test {i + 1}: {test_case} → {result} (expected {expected})")
            all_passed = False

    if all_passed:
        logger.info("✅ All classification tests passed!")
    else:
        logger.error("❌ Some classification tests failed!")

    return all_passed


if __name__ == "__main__":
    logger.info("Starting preprocessing tests...")

    # Test classification function
    test1_passed = test_classification_function()

    # Test full preprocessing pipeline
    test2_passed = test_preprocessing()

    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! Ready for training.")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed. Please check the configuration.")
        sys.exit(1)
