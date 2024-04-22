import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def fill_missing_values(df, method, neighbors = None, identifier_columns = None, data_columns = None, **kwargs):
    # TODO: fill missing values
    if method == 'zero':
        df.fillna(0, inplace = True)
        return df

    # This function will fill in missing values using the KNN algorithm. Assumes the dataset is already cleaned.

    # Parameters
    # neighbors -> number of neighbors to compare to for KNN algorithm: should be (1, 20) inclusive
    # identifier_columns -> list of column headers related to the athlete's identity (index, ID, name)
    # data_columns -> list of column headers that contain athletes' data
    
    if method == "knn":
        if (not isinstance(neighbors, int)) or neighbors <= 0 or neighbors > 20:
            raise Exception("Invalid neighbor argument")

        if not isinstance(identifier_columns, list) or len(identifier_columns) == 0:
            raise Exception("Invalid identifier_columns argument")
    
        if not isinstance(data_columns, list) or len(data_colums) == 0:
            raise Exception("Invalid data_columns argument")
    
        try:
            df_identifiers = df[identifier_columns]
            df_modify = df[data_columns]
        except KeyError:
            print(f"A column header in identifier_columns or data_columns is not in {df}.")
        else:
            df_modify = df_modify.fillna(value=np.nan)
            df_modify = df_modify[:].values
    
            imputer = KNNImputer(n_neighbors=neighbors)
    
            df_KNN = pd.DataFrame(imputer.fit_transform(df_modify), columns=data_columns)
    
            df_KNN = pd.concat([df_identifiers, df_KNN], axis=1)
    
            return df_KNN


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
