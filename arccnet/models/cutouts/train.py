import os
import argparse

import torch
from comet_ml import Experiment

import arccnet.models.cutouts.config as config
import arccnet.models.dataset_utils as ut_d
import arccnet.models.train_utils as ut_t
import arccnet.visualisation.utils as ut_v


def run_training(config, args):
    """
    Runs the full training pipeline for a machine learning model, including
    configuration setup, data preparation, model training, and logging.

    Parameters
    ----------
    config : object
        Configuration object with default settings for the training process.
    args : Namespace
        Parsed command-line arguments that may override configuration defaults.

    Returns
    -------
    tuple
        A tuple containing:
        - avg_test_loss : float
            Average test loss after model training.
        - test_accuracy : float
            Test set accuracy.
        - test_precision : float
            Precision score on the test set.
        - test_recall : float
            Recall score on the test set.
        - test_f1 : float
            F1 score on the test set.
        - cm_test : ndarray
            Confusion matrix for the test set predictions.
        - report_df : DataFrame
            Dataframe containing detailed classification report for test set.
    """

    # Override config settings with arguments if provided
    arg_to_config = {
        "model_name": "model_name",
        "batch_size": "batch_size",
        "num_epochs": "num_epochs",
        "patience": "patience",
        "learning_rate": "learning_rate",
        "gpu_indexes": "gpu_indexes",
        "data_folder": "data_folder",
        "dataset_folder": "dataset_folder",
        "df_file_name": "df_file_name",
        "num_workers": "num_workers",
    }

    for arg, attr in arg_to_config.items():
        value = getattr(args, arg)
        if value is not None:
            setattr(config, attr, value)

    if args.gpu_indexes is not None:
        config.device = (
            f"cuda:{args.gpu_indexes[0]}" if len(args.gpu_indexes) == 1 else [f"cuda:{idx}" for idx in args.gpu_indexes]
        )

    # Generate run ID and weights directory
    run_id, weights_dir = ut_t.generate_run_id(config)

    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    # Initialize Comet experiment
    run_comet = Experiment(project_name=config.project_name, workspace="arcaff")
    run_comet.add_tags([config.model_name])
    run_comet.log_parameters(
        {
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "GPU": f"GPUs_{args.gpu_indexes}" if torch.cuda.is_available() else "CPU",
            "num_epochs": config.num_epochs,
            "patience": config.patience,
        }
    )
    run_comet.log_code(config.__file__)
    run_comet.log_code(ut_t.__file__)

    # Data preparation
    print("Making dataframe...")
    df, AR_df = ut_d.make_dataframe(config.data_folder, config.dataset_folder, config.df_file_name)

    # Undersample and filter the dataframe
    df, df_du = ut_d.undersample_group_filter(
        df, config.label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1
    )

    # Split data into folds for cross-validation
    fold_df = ut_d.split_data(df_du, label_col="grouped_labels", group_col="number", random_state=42)
    df = ut_d.assign_fold_sets(df, fold_df)
    print("Dataframe preparation done.")

    # Start training
    print("Starting Training...")
    (avg_test_loss, test_accuracy, test_precision, test_recall, test_f1, cm_test, report_df) = ut_t.train_model(
        config, df, weights_dir, experiment=run_comet, use_multi_gpu=len(args.gpu_indexes) > 1
    )

    # Logging and saving assets
    print("Logging assets...")
    script_dir = os.path.dirname(ut_t.__file__)
    save_path = os.path.join(script_dir, "temp", "working_dataset.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist

    # Create directory for saving the dataset histogram if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create and log the dataset histogram
    ut_v.make_classes_histogram(
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
    parser.add_argument("--gpu_indexes", type=int, nargs="+", help="Indexes of the GPUs to use.")
    parser.add_argument("--data_folder", type=str, help="Path to the data folder.")
    parser.add_argument("--dataset_folder", type=str, help="Path to the dataset folder.")
    parser.add_argument("--df_file_name", type=str, help="Name of the dataframe file.")

    # Parse arguments
    args = parser.parse_args()

    # Call the run_training function
    run_training(config, args)
