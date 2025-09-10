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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.data_preparation import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# %%
# Define constants for column names and other magic strings
N_SPLITS = config.N_FOLDS
RANDOM_STATE = config.RANDOM_STATE

PROCESSED_DATA_PATH = (
    Path(config.DATA_FOLDER) / f"processed_dataset_{config.classes}_{N_SPLITS}-splits_rs-{RANDOM_STATE}.parquet"
)

LABEL_COL = "grouped_labels"
DATE_COL = "dates"
GROUP_COL = "number"
SPLITS = ["train", "val", "test"]
FOLD_PREFIX = "Fold"


def load_or_prepare_data(path: Path) -> pd.DataFrame:
    """Load the processed dataset if it exists, otherwise prepare it."""
    if not path.exists():
        logging.info("Preparing dataset for the first time...")
        return prepare_dataset(
            save_path=str(path),
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            label_mapping=config.label_mapping,
        )
    else:
        logging.info("Loading previously prepared dataset...")
        return pd.read_parquet(str(path))


df_processed = load_or_prepare_data(PROCESSED_DATA_PATH)


# %%
# Compute class distributions for each fold
def compute_class_distributions(df, label_col=LABEL_COL):
    """Compute class distributions with counts and percentages for all folds"""

    # Get all fold columns
    fold_columns = [col for col in df.columns if "Fold" in col]

    results = []
    for fold_col in fold_columns:
        # Calculate total samples per split for this fold
        fold_totals = {}
        total_fold_samples = 0
        split_counts = {}
        for split in ["train", "val", "test"]:
            split_counts[split] = len(df[df[fold_col] == split])
            fold_totals[split] = split_counts[split]
            total_fold_samples += split_counts[split]
        # Calculate split percentages
        split_percentages = {
            split: (count / total_fold_samples * 100) if total_fold_samples > 0 else 0
            for split, count in fold_totals.items()
        }
        for split in ["train", "val", "test"]:
            split_count = split_counts[split]
            split_label = f"{split} ({split_count}, {split_percentages[split]:.1f}%)"
            subset_df = df[df[fold_col] == split]
            if len(subset_df) > 0:
                class_counts = subset_df[label_col].value_counts().sort_index()
                class_percentages = (class_counts / len(subset_df) * 100).round(2)
                for class_name in class_counts.index:
                    count = class_counts[class_name]
                    percentage = class_percentages[class_name]
                    results.append(
                        {
                            "Fold": fold_col,
                            "Split": split_label,
                            "Class": class_name,
                            "Count": count,
                            "Percentage": percentage,
                            "Count_with_Percentage": f"{count} ({percentage}%)",
                        }
                    )
            else:
                # If no data for this split, still add a row for completeness
                results.append(
                    {
                        "Fold": fold_col,
                        "Split": split_label,
                        "Class": None,
                        "Count": 0,
                        "Percentage": 0.0,
                        "Count_with_Percentage": "0 (0.0%)",
                    }
                )
    return pd.DataFrame(results)


def create_distribution_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Computes and pivots class distribution data for display."""
    class_distributions = compute_class_distributions(df)

    # Ensure splits are ordered as train, val, test
    split_order = SPLITS
    # Extract split base name
    class_distributions["Split_base"] = class_distributions["Split"].str.extract(r"^(\w+)")
    class_distributions["Split_base"] = pd.Categorical(
        class_distributions["Split_base"], categories=split_order, ordered=True
    )
    class_distributions = class_distributions.sort_values(["Fold", "Split_base"])
    class_distributions = class_distributions.drop(columns=["Split_base"])

    # Set Split as categorical with correct order for pivot
    class_distributions["Split"] = class_distributions["Split"].astype(str)
    class_distributions["Split"] = pd.Categorical(
        class_distributions["Split"],
        categories=[s for split in split_order for s in class_distributions["Split"].unique() if s.startswith(split)],
        ordered=True,
    )

    pivot_combined = class_distributions.pivot_table(
        index=["Fold", "Split"],
        columns="Class",
        values=["Count_with_Percentage"],
        fill_value="0 (0.0%)",
        aggfunc="first",
        observed=False,  # Explicitly set to silence FutureWarning
    )
    return pivot_combined


# Compute distributions and create pivot table
pivot_table = create_distribution_pivot_table(df_processed)

print("\nCombined Split Summary and Counts by Fold, Split (with %), and Class:")
print("=" * 70)
pivot_table

# %%
# Create summary table with train/val/test split percentages
## The split summary table is now merged into the main table above, so this code is no longer needed.


# %%
# Compact plot function
def plot_class_distributions_by_fold(df, label_col=LABEL_COL):
    """Create compact subplots showing class distributions for each fold and split"""
    fold_columns = [col for col in df.columns if FOLD_PREFIX in col]
    splits = SPLITS
    n_folds = len(fold_columns)

    fig, axes = plt.subplots(n_folds, 3, figsize=(12, 3 * n_folds))
    if n_folds == 1:
        axes = axes.reshape(1, -1)

    # Consistent colors
    all_classes = sorted(df[label_col].unique())
    color_map = dict(zip(all_classes, plt.cm.Set3(range(len(all_classes)))))

    for fold_idx, fold_col in enumerate(fold_columns):
        for split_idx, split in enumerate(splits):
            ax = axes[fold_idx, split_idx]
            subset_df = df[df[fold_col] == split]

            if len(subset_df) > 0:
                counts = subset_df[label_col].value_counts().sort_index()
                percentages = (counts / len(subset_df) * 100).round(1)

                # Plot bars with labels
                ax.bar(range(len(counts)), counts.values, color=[color_map[cls] for cls in counts.index])
                for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
                    ax.text(
                        i, count + max(counts.values) * 0.02, f"{count}\n({pct}%)", ha="center", va="bottom", fontsize=8
                    )

                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
                ax.set_ylim(0, max(counts.values) * 1.2)
            else:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)

            ax.set_title(f"{fold_col} - {split.upper()}\n(n={len(subset_df)})", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.suptitle("Class Distributions by Fold and Split", fontsize=14, fontweight="bold", y=1.02)
    return fig


# Create the plot
fig = plot_class_distributions_by_fold(df_processed)
plt.show()


# %%
# KFold Split Visualization
def plot_kfold_splits(df, date_col=DATE_COL):
    """Create a KFold split visualization using vertical line markers"""
    fold_columns = [col for col in df.columns if FOLD_PREFIX in col]

    plt.figure(figsize=(18, 6))

    MARKER_WIDTH = 0.1  # Width of the vertical line markers
    MARKER_HEIGHT = 25  # Length of the vertical line markers

    # Define colors and legend tracking
    split_colors = {"train": "blue", "val": "red", "test": "gold"}
    added_legend = {split: False for split in SPLITS}

    for fold_idx, fold_col in enumerate(fold_columns, 1):
        for split in SPLITS:
            split_set = df[df[fold_col] == split]
            if not split_set.empty:
                label = split.capitalize() if not added_legend[split] else None
                plt.plot(
                    split_set[date_col].values,
                    [fold_idx] * len(split_set),
                    linestyle="none",
                    marker="_",
                    color=split_colors[split],
                    markersize=MARKER_WIDTH,
                    markeredgewidth=MARKER_HEIGHT,
                    label=label,
                    alpha=1,
                )
                if label:
                    added_legend[split] = True

    # Add title and formatting
    plt.title("KFold Split Visualization", fontsize=16, fontweight="bold")
    plt.yticks(
        ticks=range(1, len(fold_columns) + 1),
        labels=[f"Fold {i}" for i in range(1, len(fold_columns) + 1)],
        fontsize=12,
    )
    plt.xlabel("Date", fontsize=12)
    plt.grid(True, alpha=0.3, axis="x")

    # Adding legend outside the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), markerscale=2, fontsize=12)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    return plt.gcf()


# Create the KFold visualization
fig_kfold = plot_kfold_splits(df_processed)
plt.show()


# %%
# Compact Group Analysis - Check for overlaps between folds using AR numbers
def analyze_group_overlaps(df, group_col=GROUP_COL):
    """Compact analysis of group overlaps between train/val/test sets across all folds"""
    fold_columns = [col for col in df.columns if FOLD_PREFIX in col]
    all_groups = set(df[group_col].unique())
    train_groups_set, val_groups_set, test_groups_set, all_test_groups = set(), set(), set(), []

    print("GROUP OVERLAP ANALYSIS")
    print("=" * 40)

    for fold_idx, fold_col in enumerate(fold_columns, 1):
        # Get unique groups for each split
        train_groups = set(df[df[fold_col] == "train"][group_col].unique())
        val_groups = set(df[df[fold_col] == "val"][group_col].unique())
        test_groups = set(df[df[fold_col] == "test"][group_col].unique())

        print(f"Fold {fold_idx}: Train({len(train_groups)}) Val({len(val_groups)}) Test({len(test_groups)})")

        # Check for overlaps within this fold
        overlaps = []
        if train_groups & val_groups:
            overlaps.append("Train-Val")
        if train_groups & test_groups:
            overlaps.append("Train-Test")
        if val_groups & test_groups:
            overlaps.append("Val-Test")

        if overlaps:
            print(f"  ⚠️  Overlaps: {', '.join(overlaps)}")
        else:
            print("  ✅ No overlaps")

        # Update cumulative sets
        train_groups_set.update(train_groups)
        val_groups_set.update(val_groups)
        test_groups_set.update(test_groups)
        all_test_groups.extend(test_groups)

    # Overall summary
    print(f"\nSUMMARY (Total groups: {len(all_groups)})")
    print("-" * 40)
    print(
        f"Train coverage: {len(train_groups_set)}/{len(all_groups)} {'✅' if len(train_groups_set) == len(all_groups) else '⚠️'}"
    )
    print(
        f"Val coverage: {len(val_groups_set)}/{len(all_groups)} {'✅' if len(val_groups_set) == len(all_groups) else '⚠️'}"
    )
    print(
        f"Test coverage: {len(test_groups_set)}/{len(all_groups)} {'✅' if len(test_groups_set) == len(all_groups) else '⚠️'}"
    )

    # Check test set uniqueness across folds
    duplicate_test_groups = set([g for g in all_test_groups if all_test_groups.count(g) > 1])
    if duplicate_test_groups:
        print(f"Test uniqueness: ⚠️  {len(duplicate_test_groups)} groups in multiple test sets")
    else:
        print("Test uniqueness: ✅ All test sets unique")

    return {
        "duplicate_test_groups": duplicate_test_groups,
        "missing_in_train": all_groups - train_groups_set,
        "missing_in_val": all_groups - val_groups_set,
        "missing_in_test": all_groups - test_groups_set,
    }


# Perform compact group overlap analysis
analysis_results = analyze_group_overlaps(df_processed, group_col="number")

# %%
