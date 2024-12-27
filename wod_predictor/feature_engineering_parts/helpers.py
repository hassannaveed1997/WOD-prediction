import pandas as pd
import numpy as np
import warnings
from sklearn.impute import KNNImputer
from ..constants import Constants as c

LB_MULTIPLIER = 2.20462
CM_MULTIPLIER = 0.393701


def fill_missing_values(df, method, **kwargs):

    # This function will fill in missing values using the KNN algorithm. Assumes the dataset is already cleaned.

    # Parameters
    # neighbors -> number of neighbors to compare to for KNN algorithm: should be (1, 20) inclusive
    # identifier_columns -> list of column headers related to the athlete's identity (index, ID, name)
    # data_columns -> list of column headers that contain athletes' data
    SUPPORTED_METHODS = ["knn", "zero", "mean", "median"]
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Method {method} is not supported for fill_missing_values. Please use one of the following: {SUPPORTED_METHODS}"
        )

    if method == "knn":
        df_filled = fill_missing_knn(df, **kwargs)
        return df_filled
    if method == "zero":
        return df.fillna(0)
    if method == "mean":
        return df.fillna(df.mean())
    if method == "median":
        return df.fillna(df.median())


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
    df = df.copy()
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


def remove_suffixes(df):
    """
    Removes suffixes from the column valus of the dataframe

    Parameters:
    ----------
    df: pd.DataFrame
        The dataframe with columns to be modified

    Returns:
    -------
    df: pd.DataFrame
        The modified dataframe with the suffixes removed
    """
    SUFFIXES_TO_REMOVE = ["lbs", "lb", "reps"]

    for col in df.columns:
        if df[col].dtype == "object":
            for suffix in SUFFIXES_TO_REMOVE:
                df[col] = df[col].str.replace(suffix, "")
    return df


def remove_scaled_workout_columns(df):
    workout_cols = [col for col in df.columns if "." in col]

    cols_to_drop = []
    for col in workout_cols:
        if col.endswith(c.foundation_tag) or col.endswith(c.scaled_tag):
            cols_to_drop.append(col)

    df.drop(cols_to_drop, axis=1, inplace=True)
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
    df = df.copy()
    if columns is None:
        columns = [
            col for col in df.columns if "." in col
        ]  # generally open workouts have a "." in the name, e.g 17.4

    mapping = {" - f": c.foundation_tag, " - s": c.scaled_tag}
    for col in columns:
        if df[col].dtype != "object":  # if the column is not a string,
            continue

        for key, tag in mapping.items():
            rows_that_contain_key = df[col].str.contains(key)

            # keep row as it is for those containing the key, but NA for others
            if rows_that_contain_key.any():
                new_col = f"{col}{tag}"
                df[new_col] = df[col].where(rows_that_contain_key)
                df[new_col] = df[new_col].str.replace(key, "")

            # remove rows from original column
            df.loc[rows_that_contain_key == 1, col] = np.nan

    return df


def _convert_single_time_cap_workout_to_reps(x, total_reps, time_cap, scale_up=False):
    """
    Here x can either be in reps or a time. For example, if the workout had a total of 100 reps and a time cap of 20 minutes,
    and the athlete finished in 15 minutes, we could do two things:
    1. Either return the reps as is, which would be 100 in the example, because the athlete finished the workout
    2. Scale up the number of reps to 100*20/15 = 133.33, which could approximate the number of reps the athlete would have finished in 20 minutes.

    Parameters:
    ----------
    x: str
        The time or reps completed by the athlete
    total_reps: int
        The total number of reps in the workout
    time_cap: int
        The time cap for the workout in minutes
    scale_up: bool
        Whether to scale up the reps or not if they didn't complete workout
    """

    time_delta_format = "00:00:00"
    x_orig = x
    x = str(x).strip()  # convert to string if not already
    try:
        if x in [np.nan, "nan", pd.NaT]:  # if missing, return nat timedelta
            return np.nan
        elif (
            ":" in x
        ):  # if x is reported as a time, athlete finished workout which needs to be converted to reps
            x = (
                time_delta_format[: len(time_delta_format) - len(x)] + x
            )  # fixes format to "00:00:00"
            # fixes format of time_cap
            time_cap = str(time_cap) + ":00"
            time_cap = (
                time_delta_format[: len(time_delta_format) - len(time_cap)] + time_cap
            )

            if scale_up and total_reps is not None:  # return total_reps * time_cap / x
                # Convert the time strings into timedelta objects
                time_cap = pd.to_timedelta(time_cap)
                x = pd.to_timedelta(x)
                # Find scaling factor using total number of seconds in each time object
                scaling_factor = time_cap.total_seconds() / x.total_seconds()
            else:  # return total_reps
                scaling_factor = 1

            return total_reps * scaling_factor

        else:  # athlete did not finish workout, they have a score in reps which can be returned as is
            return int(x)
    except Exception as e:
        warnings.warn(
            f"Could not convert {x_orig} to reps, returning as is. Error: {e}"
        )


def _convert_single_time_cap_workout_to_time(
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
        The time cap for the workout in minutes
    scale_up: bool
        Whether to scale up the time or not if they didn't complete workout
    """

    time_delta_format = "00:00:00"
    x_orig = x
    x = str(x).strip()  # convert to string if not already
    try:
        if x in [np.nan, "nan", pd.NaT, "--"]:  # if missing, return nat timedelta
            return np.nan

        elif (
            ":" in x
        ):  # if x is reported as a time, athlete finished. Hacky Solution: Looking for ":" in the string to determine if it is a time
            # fix formatting issues
            x = time_delta_format[: len(time_delta_format) - len(x)] + x

            x = pd.to_timedelta(x)
            return x

        else:  # athlete did not finish workout, they have a score in reps which needs to be converted to time
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


def convert_time_cap_workout_to_time(
    x: pd.Series, total_reps: int, time_cap: int, scale_up=False
):
    """
    wrapper around single function, refer to helper
    """
    x_as_time =  x.apply(
        lambda x: _convert_single_time_cap_workout_to_time(
            x,
            total_reps,
            time_cap,
            scale_up=scale_up,
        )
    )
    x_as_float = x_as_time.dt.seconds/60
    return x_as_float


def convert_time_cap_workout_to_reps(
    x: pd.Series, total_reps: int, time_cap: int, scale_up=False
):
    """
    wrapper around single function, refer to helper
    """
    return x.apply(
        lambda x: _convert_single_time_cap_workout_to_reps(
            x,
            total_reps,
            time_cap,
            scale_up=scale_up,
        )
    )


def convert_rpm(x: pd.Series, total_reps, time_cap, scale_up=True):
    """
    Converts a mixed column into floats
    """
    if scale_up == False:
        warnings.warn(
            "Using RPM method with scale up = False. Set to True for best results"
        )
    pd.set_option("future.no_silent_downcasting", True)  # to silence a warning
    rpm = pd.Series(index=x.index, dtype=float)

    missing_index = x.isnull()
    is_timed = x.str.contains(":").fillna(False).astype(bool)
    timed_indices = x.loc[is_timed & ~missing_index].index
    reps_indices = x.loc[~is_timed & ~missing_index].index
    max_reps = 0
    if len(reps_indices) > 0:
        numeric_results = pd.to_numeric(x.loc[reps_indices], errors="coerce")
        max_reps = max(numeric_results)
        rpm_from_reps = numeric_results / time_cap
        rpm.loc[reps_indices] = rpm_from_reps.astype(float)

    if len(timed_indices) > 0:
        if time_cap is None:
            raise ValueError("Timecap must be provided for workouts that are for time")
        if max_reps > total_reps:
            raise Warning(
                f"Timecap must be greater than the maximum reps in the dataset, please sanity check the descriptions and data. Workout {x.name} has observed max reps: {max_reps}, but given total reps: {total_reps}"
            )
        # apply the time conversion function
        times = x.loc[timed_indices].apply(
            lambda x: _convert_single_time_cap_workout_to_time(
                x, total_reps, time_cap, scale_up=scale_up
            )
        )
        times_in_minutes = times.dt.total_seconds() / 60
        rpm_from_time = total_reps / times_in_minutes

        rpm.loc[timed_indices] = rpm_from_time.astype(float)

    return rpm


def convert_to_floats(
    df: pd.DataFrame,
    descriptions: dict,
    conversion_method="rpm",
    scale_up=True,
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
    conversion_method: str
        Allows multiple conversion methods for mixed data types. Options are "to_time", "to_reps", "rpm" (reps per minute).
    Returns:
    -------
    df_modified: pd.DataFrame
        The modified dataframe with the scores for the workout as integers. This would either be reps, or time in minutes.
        These can be added as new columns. For example, 17.1 would become 17.1_reps if it was a workout for reps.
    """
    df_modified = df.copy()
    df_modified.columns = [col.replace("_score", "") for col in df_modified.columns]
    df_modified.replace("--", "", inplace=True)

    # determine function based on goal
    func_by_goal_mapping = {
        "load": {"function": pd.to_numeric, "kwargs": {"errors": "ignore"}},
        "amrap": {"function": pd.to_numeric, "kwargs": {"errors": "ignore"}},
        "for time": {"function": pd.to_numeric, "kwargs": {"errors": "ignore"}},
    }

    match conversion_method:
        case "rpm":
            func_by_goal_mapping["amrap"] = {"function": convert_rpm}
            func_by_goal_mapping["for time"] = {"function": convert_rpm}
        case "to_reps":
            func_by_goal_mapping["for time"] = {
                "function": convert_time_cap_workout_to_reps
            }
        case "to_time":
            func_by_goal_mapping["for time"] = {
                "function": convert_time_cap_workout_to_time
            }

    for workout_name in df_modified.columns:
        workout_name_base = _get_workout_base_name(workout_name, descriptions)

        goal = descriptions[workout_name_base]["goal"].lower()
        funct_to_apply = func_by_goal_mapping[goal]["function"]
        kwargs = func_by_goal_mapping[goal].get("kwargs")
        if kwargs is None:  # set kwargs if not available
            kwargs = {
                "total_reps": descriptions[workout_name_base]["total_reps"],
                "time_cap": descriptions[workout_name_base]["time_cap"],
                "scale_up": scale_up,
            }

        df_modified[workout_name] = funct_to_apply(df_modified[workout_name], **kwargs)

        # manual inspection
        try:
            pd.to_numeric(df_modified[workout_name])
        except Exception as e:
            warnings.warn(
                f"{e}: Could not convert column {workout_name} to float. Please inspect manually"
            )
    # final cast of dtypes
    df_modified = df_modified.astype(float)

    return df_modified


def _get_workout_base_name(workout_name, descriptions):
    # remove any suffixes from the workout name for easy lookup
    workout_name_base = workout_name.replace(c.scaled_tag, "").replace(
        c.foundation_tag, ""
    )

    # if we don't have the workout name in descriptions, we skip it
    if workout_name_base not in descriptions:
        raise ValueError(
            f"Workout {workout_name} not found in descriptions, skipping preprocessing. Please inspect manually"
        )
    return workout_name_base
