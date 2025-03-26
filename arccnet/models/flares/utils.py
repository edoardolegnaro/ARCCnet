import pandas as pd

def add_flare_flag(df, threshold='C'):
    """
    Adds a 'flare_flag' column to the DataFrame based on the 'flare_class' column.
    The 'flare_flag' indicates whether the flare class is at or above the specified threshold.

    Args:
        df (pd.DataFrame): The DataFrame containing the 'flare_class' column.
        threshold (str): The flare class threshold ('A', 'C', 'M', or 'X').
                         'A' means all flares are considered, 'C' means C-class or higher, etc.
                         Defaults to 'C'.

    Returns:
        pd.DataFrame: The DataFrame with the added 'flare_flag' column.

    Raises:
        ValueError: If an invalid threshold is provided.
    """

    valid_thresholds = ['A', 'B', 'C', 'M', 'X']
    if threshold not in valid_thresholds:
        raise ValueError(f"Invalid threshold: '{threshold}'. Must be one of {valid_thresholds}")

    df['flare_flag'] = False  # Initialize the new column with False

    for index, row in df.iterrows():
        flare_class = row['flare_class']

        if isinstance(flare_class, str) and len(flare_class) > 0:
            flare_prefix = flare_class[0].upper()  # Ensure uppercase for comparison

            if threshold == 'A':
                df.loc[index, 'flare_flag'] = True  # All flares are flagged as True
            elif threshold == 'C':
                if flare_prefix in ['C', 'M', 'X']:
                    df.loc[index, 'flare_flag'] = True
            elif threshold == 'M':
                if flare_prefix in ['M', 'X']:
                    df.loc[index, 'flare_flag'] = True
            elif threshold == 'X':
                if flare_prefix == 'X':
                    df.loc[index, 'flare_flag'] = True

    return df
