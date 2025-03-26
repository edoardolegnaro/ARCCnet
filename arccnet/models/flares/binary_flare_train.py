# binary_flare_train.py
import os
import pandas as pd
import numpy as np
import argparse
from comet_ml import Experiment

import torch
import arccnet.models.cutouts.config as base_config
import arccnet.models.train_utils as ut_t
import arccnet.models.dataset_utils as ut_d


class FlareConfig:
    """Configuration class for flare prediction model training."""
    
    def __init__(self):
        # Copy relevant settings from base config
        for attr in dir(base_config):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(base_config, attr))
        
        # Override with flare-specific settings
        self.model_name = "resnet18"
        self.batch_size = 32
        self.num_epochs = 50
        self.patience = 10
        self.learning_rate = 1e-4
        self.num_workers = 4
        self.pretrained = True
        self.project_name = "ARs-Flare-Classification"
        self.other_tags = ["binary", "flare-prediction"]
        
        # Binary classification specific
        self.num_classes = 2
        self.label_mapping = {'quiet': 0, 'flares': 1}


def prepare_flare_dataset(data_folder, dataset_folder, flare_threshold='C'):
    """Prepare the flare classification dataset."""
    # Load flare data
    df_file_name = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"
    df_flares = pd.read_parquet(os.path.join(data_folder, df_file_name))
    
    # Check file existence
    df_flares_exists = check_fits_file_existence(df_flares.copy(), data_folder, dataset_folder)
    df_flares_data = df_flares_exists[df_flares_exists['file_exists']]
    
    # Apply binary classification labels
    df_flares_data['flaring_flag'] = df_flares_data.apply(
        lambda row: categorize_flare(row, threshold=flare_threshold), axis=1
    )
    
    # Add required columns for training infrastructure
    df_flares_data['grouped_labels'] = df_flares_data['flaring_flag']
    df_flares_data['encoded_labels'] = df_flares_data['flaring_flag'].map({'quiet': 0, 'flares': 1})
    
    return df_flares_data


def categorize_flare(row, threshold='C'):
    """Categorize a row as 'flares' or 'quiet' based on flare levels."""
    flare_levels = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
    if threshold not in flare_levels:
        raise ValueError("Invalid threshold. Must be 'A', 'B', 'C', 'M', or 'X'.")

    threshold_value = flare_levels[threshold]

    for level, value in flare_levels.items():
        if value >= threshold_value and not pd.isna(row[level]):
            return 'flares'
    return 'quiet'


def check_fits_file_existence(df, data_folder, dataset_folder):
    """Check if FITS files exist and mark them in the dataframe."""
    df['file_exists'] = False

    for index, row in df.iterrows():
        if row["path_image_cutout_hmi"] is not None:
            path_key = "path_image_cutout_hmi"
        elif row["path_image_cutout_mdi"] is not None:
            path_key = "path_image_cutout_mdi"
        else:
            continue

        if row[path_key] is None or not isinstance(row[path_key], str):
            continue

        fits_file_path = os.path.join(data_folder, dataset_folder, 'fits', os.path.basename(row[path_key]))
        if os.path.exists(fits_file_path):
            df.loc[index, 'file_exists'] = True

    return df


def split_dataframe(df, test_size=0.1, val_size=0.2, random_state=42):
    """Split dataframe into train, validation, and test sets."""
    from sklearn.model_selection import GroupShuffleSplit
    
    # First split into train+val and test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['number']))
    
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]
    
    # Split train+val into train and validation
    adjusted_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['number']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    # Create fold columns needed by training infrastructure
    df['Fold 1'] = 'none'  # Initialize
    df.loc[df.index.isin(train_df.index), 'Fold 1'] = 'train'
    df.loc[df.index.isin(val_df.index), 'Fold 1'] = 'val'
    df.loc[df.index.isin(test_df.index), 'Fold 1'] = 'test'
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flare classification training script.")
    parser.add_argument("--model_name", type=str, help="Model architecture (from timm)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--patience", type=int, help="Patience for early stopping")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--gpu_index", type=int, help="GPU index")
    parser.add_argument("--flare_threshold", type=str, default="C", 
                        help="Flare threshold (A, B, C, M, X)")
    
    args = parser.parse_args()
    
    # Initialize custom config
    flare_config = FlareConfig()
    
    # Override config with command line args
    for arg_name, value in vars(args).items():
        if value is not None and hasattr(flare_config, arg_name):
            setattr(flare_config, arg_name, value)
    
    # Set device based on GPU index if provided
    if args.gpu_index is not None:
        flare_config.device = f"cuda:{args.gpu_index}"
    else:
        flare_config.device = ut_t.get_device()
    
    # Prepare dataset
    print("Preparing flare classification dataset...")
    df = prepare_flare_dataset(
        flare_config.data_folder, 
        flare_config.dataset_folder,
        args.flare_threshold or "C"
    )
    
    # Split into train/val/test sets
    print("Splitting dataset into train/val/test sets...")
    df = split_dataframe(df)
    
    # Generate run ID and weights directory
    run_id, weights_dir = ut_t.generate_run_id(flare_config)
    
    # Initialize Comet experiment
    print("Setting up experiment tracking...")
    experiment = Experiment(project_name=flare_config.project_name, workspace="arcaff")
    experiment.add_tags([flare_config.model_name, f"threshold-{args.flare_threshold or 'C'}"])
    experiment.log_parameters({
        "model_name": flare_config.model_name,
        "batch_size": flare_config.batch_size,
        "num_epochs": flare_config.num_epochs,
        "flare_threshold": args.flare_threshold or "C",
    })
    
    # Start training
    print(f"Starting training with {flare_config.model_name}...")
    (avg_test_loss, test_accuracy, test_precision, 
     test_recall, test_f1, cm_test, report_df) = ut_t.train_model(
        flare_config, df, weights_dir, experiment=experiment
    )
    
    print(f"Training complete. Test accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")