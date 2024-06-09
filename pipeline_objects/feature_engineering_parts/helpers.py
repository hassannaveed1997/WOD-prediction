import pandas as pd
import numpy as np
import warnings
from sklearn.impute import KNNImputer

LB_MULTIPLIER = 2.20462
CM_MULTIPLIER = 0.393701


def fill_missing_values(df, method, **kwargs):

    # This function will fill in missing values using the KNN algorithm. Assumes the dataset is already cleaned.

    # Parameters
    # neighbors -> number of neighbors to compare to for KNN algorithm: should be (1, 20) inclusive
    # identifier_columns -> list of column headers related to the athlete's identity (index, ID, name)
    # data_columns -> list of column headers that contain athletes' data
    SUPPORTED_METHODS = ["knn", "zero"]
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method {method} is not supported for fill_missing_values. Please use one of the following: {SUPPORTED_METHODS}"
        )

    if method == "knn":
        df_filled = fill_missing_knn(df, **kwargs)
        return df_filled
    if method == "zero":
        return df.fillna(0)


def fill_missing_knn(df, neighbors, data_columns=[]):
    """
    This function will fill in missing values using the KNN algorithm. Assumes outliers are removed.
    """
    if (not isinstance(neighbors, int)) or neighbors <= 0 or neighbors > 20:
        raise Exception("Invalid neighbor argument")

    if not isinstance(data_columns, list):
        raise Exception("Invalid data_columns argument")

    if len(data_columns) == 0:
        # include numeric columns only
        data_columns = df.select_dtypes(include=["int", "float"]).columns

    # must have at least one non-missing value in the column
    data_columns = [col for col in data_columns if df[col].notnull().sum() > 0]

    try:
        df_identifiers = df.drop(columns=data_columns)
        df_modify = df[data_columns]
    except KeyError as e:
        raise KeyError(
            f"A column header in identifier_columns or data_columns is not in df: {e}"
        )

    df_modify = df_modify.fillna(value=np.nan)
    df_modify = df_modify[:].values

    imputer = KNNImputer(n_neighbors=neighbors)
    df_KNN = imputer.fit_transform(df_modify)
    df_KNN = pd.DataFrame(df_KNN, columns=data_columns, index=df.index)

    df_KNN = pd.concat([df_identifiers, df_KNN], axis=1)
    return df_KNN


def remove_outliers(df, method="iqr", score_headers: list = None, **kwargs):
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
    SUPPORTED_METHODS = ["iqr"]
    if method not in SUPPORTED_METHODS:  # add any more methods here
        raise ValueError(
            f"Method {method} is not supported for outlier detection. Please use one of the following: {SUPPORTED_METHODS}"
        )

    df_modified = df.copy()

    # If score_headers is None, use all columns thst are numeric
    if score_headers is None:
        score_headers = df.select_dtypes(include=["int", "float"]).columns

    # Sanity check to confirm that the columns are numeric
    for col in score_headers:
        if df_modified[col].dtype not in [int, float]:
            raise ValueError(f"Column {col} is not numeric, convert to numeric first")

    # fill 0 with na first to prevent skewing the results
    df_modified = df_modified.replace(0, np.nan)

    # find interquartile range
    if method == "iqr":
        upper_quartiles = df_modified[score_headers].quantile(0.75)
        lower_quartiles = df_modified[score_headers].quantile(0.25)
        iqr = upper_quartiles - lower_quartiles

        # find outliers
        outliers = (df_modified[score_headers] > (upper_quartiles + 1.5 * iqr)) | (
            df_modified[score_headers] < (lower_quartiles - 1.5 * iqr)
        )
        df_modified[outliers] = np.nan

    return df_modified


def convert_units(df, type, columns=None):
    """
    Helper function to convert units from metric to imperial

    Parameters:
    ----------
    df: pd.DataFrame
        The dataframe with the columns to be converted
    type: str
        The type of conversion. It can be either 'weight' or 'height'
    columns: list
        The columns to be converted. If None, all columns would be tested except 'name'

    Returns:
    -------
    df: pd.DataFrame
        The modified dataframe with the converted columns

    """
    # if no columns are provided, use all columns except 'name'
    if not columns:
        columns = list(df.columns)
        if "name" in columns:
            columns.remove("name")

    # determine the multiplier and key words to remove based on the type
    if type == "weight":
        multiplier = LB_MULTIPLIER
        key_words_to_remove = ["kg", "lb"]

    elif type == "height":
        multiplier = CM_MULTIPLIER
        key_words_to_remove = ["cm", "in"]
    else:
        raise ValueError('type must be "weight" or "height"')

    # iterate through each column and convert the units
    for col in columns:
        if df[col].dtype == "object":  # if its not an object, we can skip
            # determine which rows contain the key words
            rows_with_key_words = df[col].str.contains(key_words_to_remove[0], na=False)
            if not rows_with_key_words.any():
                continue
            for word in key_words_to_remove:
                df[col] = df[col].str.replace(word, "", regex=False)
            df[col] = df[col].str.strip()

            df[col] = df[col].astype(float)
            df.loc[rows_with_key_words, col] = (
                df.loc[rows_with_key_words, col] * multiplier
            )
            print(f"Converted {col} to {type} in imperial units")
    return df


def seperate_scaled_workouts(df, columns=None):
    """
    Seperates the scaled and foundation workouts into different columns, to be treated as different workouts

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the workouts
    columns : list
        The columns to be treated as workouts. If None, the function will look for columns with a "." in the name

    Returns
    -------
    pd.DataFrame
        The dataframe with additional columns for the scaled and foundation workouts
    """
    df.copy()
    if columns is None:
        columns = [
            col for col in df.columns if "." in col
        ]  # generally open workouts have a "." in the name, e.g 17.4
    mapping = {" - f": "foundation", " - s": "scaled"}
    for col in columns:
        if df[col].dtype != "object":  # if the column is not a string,
            continue

        df[col] = df[col].str.replace("lbs", "")  # in case there are lbs in the column
        df[col] = (
            df[col].str.replace("reps", "").str.strip()
        )  # if there were reps in column

        for key, value in mapping.items():
            rows_that_contain_key = df[col].str.contains(key)

            # keep row as it is for those containing the key, but NA for others
            if rows_that_contain_key.any():
                new_col = f"{col}_{value}"
                df[new_col] = df[col].where(rows_that_contain_key)
                df[new_col] = df[new_col].str.replace(key, "")

            # remove rows from original column
            df.loc[rows_that_contain_key == 1, col] = np.nan

    return df


def convert_time_cap_workout_to_reps(x, total_reps, time_cap, scale_up=False):
    """
    Here x can either be in reps or a time. For example, if the workout had a total of 100 reps and a time cap of 20 minutes,
    and the athlete finished in 15 minutes, we could do two things:
    1. Either return the reps as is, which would be 100 in the example, because the athlete finished the workout
    2. Scale up the number of reps to 100*20/15 = 133.33, which could approximate the number of reps the athlete would have finished in 20 minutes.
    """
    # TODO: Finish this function
    raise NotImplementedError
    # if x is reported as reps
    if isinstance(x, int):
        # athlete finished the workout
        if x >= total_reps:
            # athlete did not finish the workout -> scale_up
            return x
        else:
            return total_reps * time_cap / x
    # if x is reported as time, then athlete finished workout
    else:
        return total_reps


def convert_time_cap_workout_to_time(
    x: str, total_reps: int, time_cap: int, scale_up=False
):
    """
    Here x can either be in reps or a time. For example, if the workout had a total of 100 reps and a time cap of 20 minutes,
    and the athlete finished only finished 80 reps by the cap, we could do two things:
    1. Either return the time as is, which would be 20 in the example, because the athlete finished the workout
    2. Scale up the time to 20*100/80 = 25, which could approximate the time the athlete would have taken finished the workout. (Not necessarily though)

    Parameters:
    ----------
    x: str
        The time or reps completed by the athlete
    total_reps: int
        The total number of reps in the workout
    time_cap: int
        The time cap for the workout in munutes
    scale_up: bool
        Whether to scale up the time or not if they didn't complete workout
    """

    time_delta_format = "00:00:00"
    x_orig = x
    x = str(x).strip()  # convert to string if not already
    try:
        if x in [np.nan, "nan", pd.NaT]:  # if missing, return nat timedelta
            return pd.NaT

        elif (
            ":" in x
        ):  # if x is reported as a time, athlete finished. Hacky Solution: Looking for ":" in the string to determine if it is a time
            # fix formatting issues
            x = time_delta_format[: len(time_delta_format) - len(x)] + x

            x = pd.to_timedelta(x)
            return x

        else:  # athlete did not finish workout, they have a score in reps which needs to be converted to reps
            time_cap = str(time_cap) + ":00"
            time_cap = (
                time_delta_format[: len(time_delta_format) - len(time_cap)] + time_cap
            )
            time_cap = pd.to_timedelta(time_cap)

            if scale_up and total_reps is not None:
                scaling_factor = total_reps / int(x)
            else:
                scaling_factor = 1

            return time_cap * scaling_factor
    except Exception as e:
        warnings.warn(
            f"Could not convert {x_orig} to time, returning as is. Error: {e}"
        )


def convert_to_floats(
    df: pd.DataFrame, descriptions: dict, threshold=0.5, scale_up=False
):
    """
    This function will take a workout from the open, such as 17.1, 17.3, etc. and convert it to a float. It is intended to be a standard way to convert workouts to floats.
    We can use descriptions from "WOD-prediction/Data/workout_descriptions/open_parsed_descriptions.json"

    Parameters:
    ----------
    df: pd.DataFrame
        The dataframe with the scores for the workout
    description: str
        The description of crossfit open workouts as a dictionary. It will contain fields such as "goal", "reps", "time_cap", etc.
    Returns:
    -------
    df_modified: pd.DataFrame
        The modified dataframe with the scores for the workout as integers. This would either be reps, or time in minutes.
        These can be added as new columns. For example, 17.1 would become 17.1_reps if it was a workout for reps.
    """
    df_modified = df.copy()
    df_modified.columns = [col.replace("_score", "") for col in df_modified.columns]
    for workout_name in df_modified.columns:
        # remove any suffixes from the workout name for easy lookup
        workout_name_base = workout_name.replace("_scaled", "").replace(
            "_foundation", ""
        )

        # if we don't have the workout name in descriptions, we skip it
        if workout_name_base not in descriptions:
            warnings.warn(
                f"Workout {workout_name} not found in descriptions, skipping preprocessing. Please inspect manually"
            )
            continue

        # if workout is for REPS, we should be fine
        if descriptions[workout_name_base]["goal"].lower() in ["reps", "amrap"]:
            df_modified[workout_name] = df_modified[workout_name].astype(
                int, errors="ignore"
            )
            # TODO: handle any edge cases here

        elif descriptions[workout_name_base]["goal"].lower() == "for time":
            if descriptions[workout_name_base]["time_cap"] is not None:
                # if the workout has a time cap, some athletes might not have finished it.
                # The ones that finished would have a time, the rest would have reps completed until the timecap.
                df_modified[workout_name] = df_modified[workout_name].apply(
                    lambda x: convert_time_cap_workout_to_time(
                        x,
                        descriptions[workout_name_base]["total_reps"],
                        descriptions[workout_name_base]["time_cap"],
                        scale_up=scale_up,
                    )
                )
                # convert the time to minutes
                df_modified[workout_name] = (
                    df_modified[workout_name].dt.total_seconds() / 60
                )
            else:
                # if there is no time cap, raise error
                raise ValueError(
                    f"Workout {workout_name} is for time, but does not have a time cap. Please inspect descriptions manually"
                )
        elif descriptions[workout_name_base]["goal"].lower() == "load":
            # remove anny "--"
            df_modified[workout_name] = df_modified[workout_name].str.replace("--", "")
            # if the workout is for load, we can convert it to a float
            df_modified[workout_name] = pd.to_numeric(df_modified[workout_name])
        else:
            raise ValueError(
                f"Workout {workout_name} has an unknown goal. Please inspect descriptions manually"
            )

        # manual inspection
        try:
            pd.to_numeric(df_modified[workout_name])
        except ValueError:
            warnings.warn(
                f"Could not convert column {workout_name} to float. Please inspect manually"
            )

    return df_modified
