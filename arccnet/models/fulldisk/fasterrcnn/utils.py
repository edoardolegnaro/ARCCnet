import os
import logging

import pandas as pd

from astropy.time import Time

img_size_dic = {"MDI": 1024, "HMI": 4096}


def clean_data(
    local_path_root: str = None,
    df_name: str = "fulldisk-detection-catalog-v20240917.parq",
    lon_threshold: float = 70.0,
    min_size: float = 0.024,
    img_size_dic: dict = {"MDI": 1024, "HMI": 4096},
) -> pd.DataFrame:
    """
    Loads and cleans the full-disk dataset.

    Args:
        local_path_root (str, optional): Root directory for data. If None, retrieves from environment variable
                                         'ARCAFF_DATA_FOLDER' or defaults to '../../../../../data/'. Defaults to None.
        df_name (str, optional): Name of the parquet file containing the dataset. Defaults to "fulldisk-detection-catalog-v20240917.parq".
        lon_threshold (float, optional): Longitude threshold for filtering. Defaults to 70.0.
        min_size (float, optional): Minimum size threshold for width and height. Defaults to 0.024.
        img_size_dic (dict, optional): Dictionary mapping instruments to image sizes. Defaults to {"MDI": 1024, "HMI": 4096}.

    Returns:
        pd.DataFrame: Cleaned DataFrame after applying all filters and calculations.
    """
    logger = logging.getLogger(__name__)
    if local_path_root is None:
        data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data/")
        dataset_folder = "arccnet-fulldisk-dataset-v20240917"
        local_path_root = os.path.join(data_folder, dataset_folder)
    logger.info(f"Using local_path_root: {local_path_root}")

    # Load the DataFrame
    df_path = os.path.join(local_path_root, df_name)
    logger.info(f"Loading DataFrame from {df_path}")
    try:
        df = pd.read_parquet(df_path)
        logger.info(f"DataFrame loaded successfully with {len(df)} records.")
    except FileNotFoundError:
        logger.error(f"Parquet file not found at {df_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Parquet file at {df_path} is empty")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the parquet file: {e}")
        raise

    # Process the 'time' and 'datetime' columns
    logger.info("Processing time columns")
    df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
    times = Time(df["time"], format="jd")
    df["datetime"] = pd.to_datetime(times.iso)
    logger.info("Time columns processed.")

    # Filter out rows where 'filtered' is True
    logger.info("Filtering out rows where 'filtered' is True")
    selected_df = df[~df["filtered"]].copy()
    logger.info(f"Selected DataFrame contains {len(selected_df)} records after filtering 'filtered' flag.")

    # Apply longitude threshold
    logger.info(f"Applying longitude threshold of ±{lon_threshold} degrees")
    front_df = selected_df[
        (selected_df["longitude"] < lon_threshold) & (selected_df["longitude"] > -lon_threshold)
    ].copy()
    logger.info(f"Front DataFrame contains {len(front_df)} records after applying longitude filter.")

    # Extract bounding box coordinates using .loc for safe assignment
    logger.info("Extracting bounding box coordinates")
    try:
        front_df.loc[:, ["x_min", "y_min"]] = pd.DataFrame(
            front_df["bottom_left_cutout"].tolist(), index=front_df.index
        )
        front_df.loc[:, ["x_max", "y_max"]] = pd.DataFrame(front_df["top_right_cutout"].tolist(), index=front_df.index)
        logger.info("Bounding box coordinates extracted successfully.")
    except Exception as e:
        logger.error(f"Error extracting bounding box coordinates: {e}")
        raise

    # Map image sizes based on the instrument
    logger.info("Mapping image sizes based on the instrument")
    front_df.loc[:, "img_size"] = front_df["instrument"].map(img_size_dic)
    missing_img_size = front_df["img_size"].isna().sum()
    if missing_img_size > 0:
        logger.warning(f"{missing_img_size} records have missing 'img_size' after mapping. These will be excluded.")
    logger.info("Image sizes mapped successfully.")

    # Calculate width and height using .loc
    logger.info("Calculating width and height")
    front_df.loc[:, "width"] = (front_df["x_max"] - front_df["x_min"]) / front_df["img_size"]
    front_df.loc[:, "height"] = (front_df["y_max"] - front_df["y_min"]) / front_df["img_size"]
    logger.info("Width and height calculated successfully.")

    # Handle missing img_size by dropping such records
    if missing_img_size > 0:
        front_df = front_df.dropna(subset=["img_size", "width", "height"])
        logger.info(f"Dropped {missing_img_size} records with missing 'img_size', 'width', or 'height'.")

    # Apply minimum size filter
    logger.info(f"Applying minimum size filter: width and height >= {min_size}")
    cleaned_df = front_df[(front_df["width"] >= min_size) & (front_df["height"] >= min_size)].copy()
    logger.info(f"Cleaned DataFrame contains {len(cleaned_df)} records after applying size filters.")

    return cleaned_df
