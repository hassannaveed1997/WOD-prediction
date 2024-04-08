import pandas as pd
import numpy as np


def fill_missing_values(df, method = 'knn', **kwargs):
    # TODO: fill missing values
    if method == 'zero':
        df.fillna(0, inplace = True)

    return df

def remove_outliers(df, method = 'iqr', score_headers: list = None,  **kwargs):
    """
    This function will detect outliers in the data. and replace them with missing values

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe

    method : str
        The method to use to detect outliers. Currently only 'iqr' is supported

    score_headers : list
        The columns to check for outliers. If None, all numeric columns will be checked

    Returns
    -------
    df_modified : pd.DataFrame
        The modified dataframe with outliers replaced with missing values
    """
    # sanity check on inputs
    if method not in ['iqr']: # add any more methods here
        raise ValueError(f"Method {method} is not supported. Please use one of the following: ['iqr']")
    
    df_modified = df.copy()

    # If score_headers is None, use all columns thst are numeric
    if score_headers is None:
        score_headers = df.select_dtypes(include = ['int', 'float']).columns
        
    # Sanity check to confirm that the columns are numeric
    for col in score_headers:
        if df_modified[col].dtype not in [int, float]:
            raise ValueError(f"Column {col} is not numeric, convert to numeric first")
    
    # fill 0 with na first to prevent missing
    df_modified = df_modified.replace(0, np.nan)

    # find interquartile range
    if method == 'iqr':
        upper_quartiles = df_modified[score_headers].quantile(0.75)
        lower_quartiles = df_modified[score_headers].quantile(0.25) 
        iqr = upper_quartiles - lower_quartiles

        # find outliers
        outliers = (df_modified[score_headers] > (upper_quartiles + 1.5 * iqr)) | (df_modified[score_headers] < (lower_quartiles - 1.5 * iqr))
        df_modified[outliers] = np.nan

    return df_modified