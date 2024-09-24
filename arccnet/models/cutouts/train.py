import os
import argparse

import torch
from comet_ml import Experiment

import arccnet.models.cutouts.config as config
import arccnet.models.utilities as ut


def run_training(config, args):
    """
    This function handles the entire training pipeline including:
    - Setting up the configuration.
    - Initializing the Comet.ml experiment.
    - Data preparation and model training.
    - Logging results and artifacts.

    Parameters:
    - config: The configuration object containing default values.
    - args: Parsed command-line arguments that override config values.
    """

    # Override config settings with arguments if provided
    arg_to_config = {
        "model_name": "model_name",
        "batch_size": "batch_size",
        "num_epochs": "num_epochs",
        "patience": "patience",
        "learning_rate": "learning_rate",
        "gpu_index": "gpu_index",
        "data_folder": "data_folder",
        "dataset_folder": "dataset_folder",
        "df_file_name": "df_file_name",
        "num_workers": "num_workers",
    }

    for arg, attr in arg_to_config.items():
        value = getattr(args, arg)
        if value is not None:
            setattr(config, attr, value)

    if args.gpu_index is not None:
        config.device = f"cuda:{args.gpu_index}"

    # Generate run ID and weights directory
    run_id, weights_dir = ut.generate_run_id(config)

    # Initialize Comet experiment
    run_comet = Experiment(project_name=config.project_name, workspace="arcaff")
    run_comet.add_tags([config.model_name])
    run_comet.log_parameters(
        {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "GPU": f"GPU{config.gpu_index}_{torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU",
            "num_epochs": config.num_epochs,
            "patience": config.patience,
        }
    )
    run_comet.log_code(config.__file__)
    run_comet.log_code(ut.__file__)

    # Data preparation
    print("Making dataframe...")
    df, AR_df = ut.make_dataframe(config.data_folder, config.dataset_folder, config.df_file_name)

    # Undersample and filter the dataframe
    df, df_du = ut.undersample_group_filter(
        df, config.label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1
    )

    # Split data into folds for cross-validation
    fold_df = ut.split_data(df_du, label_col="grouped_labels", group_col="number", random_state=42)
    df = ut.assign_fold_sets(df, fold_df)
    print("Dataframe preparation done.")

    # Start training
    print("Starting Training...")
    (avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df) = ut.train_model(
        config, df, weights_dir, experiment=run_comet
    )

    # Logging and saving assets
    print("Logging assets...")
    script_dir = os.path.dirname(ut.__file__)
    save_path = os.path.join(script_dir, "temp", "working_dataset.png")

    # Create and log the dataset histogram
    ut.make_classes_histogram(
        df_du["grouped_labels"], title="Dataset (Grouped Undersampled)", y_off=100, figsz=(7, 5), save_path=save_path
    )
    run_comet.log_image(save_path)

    # Log the dataset as a CSV
    run_comet.log_asset_data(df.to_csv(index=False), name="dataset.csv")

    print("Training complete.")
    return avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Training script with configurable options.")
    parser.add_argument("--model_name", type=str, help="Timm model name")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, help="Number of workers for data loading and preprocessing.")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs for training.")
    parser.add_argument("--patience", type=int, help="Patience for early stopping.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for optimizer.")
    parser.add_argument("--gpu_index", type=int, help="Index of the GPU to use.")
    parser.add_argument("--data_folder", type=str, help="Path to the data folder.")
    parser.add_argument("--dataset_folder", type=str, help="Path to the dataset folder.")
    parser.add_argument("--df_file_name", type=str, help="Name of the dataframe file.")

    # Parse arguments
    args = parser.parse_args()

    # Call the run_training function
    run_training(config, args)
