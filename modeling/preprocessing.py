import pandas as pd
import numpy as np

LB_MULTIPLIER = 2.20462
CM_MULTIPLIER = 0.393701

def convert_units(df, type, columns = None):
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
        if 'name' in columns:
            columns.remove('name')

    # determine the multiplier and key words to remove based on the type
    if type == 'weight':
        multiplier = LB_MULTIPLIER
        key_words_to_remove = ['kg', 'lb']

    elif type == 'height':
        multiplier = CM_MULTIPLIER
        key_words_to_remove = ['cm', 'in']
    else:
        raise ValueError('type must be "weight" or "height"')

    # iterate through each column and convert the units
    for col in columns:
        if df[col].dtype == 'object': # if its not an object, we can skip
            # determine which rows contain the key words
            rows_with_key_words = df[col].str.contains(key_words_to_remove[0], na = False) 
            if not rows_with_key_words.any():
                continue
            for word in key_words_to_remove:
                df[col] = df[col].str.replace(word, '', regex = False)
            df[col] = df[col].str.strip()


            df[col] = df[col].astype(float)
            df.loc[rows_with_key_words,col] = df.loc[rows_with_key_words,col] * multiplier
            print(f"Converted {col} to {type} in imperial units")
    return df


def seperate_scaled_workouts(df, columns = None):
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
        columns = [col for col in df.columns if "." in col] # generally open workouts have a "." in the name, e.g 17.4
    mapping = {" - f": "foundation", " - s": "scaled"}
    for col in columns:
        if df[col].dtype != "object": # if the column is not a string,
            continue

        df[col] = df[col].str.replace("lbs", "") # in case there are lbs in the column
        df[col] = df[col].str.replace("reps", "").str.strip() # if there were reps in column

        for key, value in mapping.items():
            rows_that_contain_key = df[col].str.contains(key)

            # keep row as it is for those containing the key, but NA for others
            if rows_that_contain_key.any():
                new_col = f'{col}_{value}'
                df[new_col] = df[col].where(rows_that_contain_key)
                df[new_col] = df[new_col].str.replace(key, "")

            # remove rows from original column
            df.loc[rows_that_contain_key==1, col] = np.nan
                    
    return df

def convert_to_datetime(dt_col):
    """
    Converts a whole column to datetime if it is of type object.
    """
    if dt_col.dtype == 'O':
        # replace str nan to np.NaN
        dt_col = dt_col.replace('nan', np.NaN)

        dt_col = pd.to_timedelta(dt_col)
        return dt_col
    return dt_col

def convert_time_cap_workout_to_reps(x, total_reps, time_cap, scale_up=False):
    """
    Here x can either be in reps or a time. For example, if the workout had a total of 100 reps and a time cap of 20 minutes,
    and the athlete finished in 15 minutes, we could do two things:
    1. Either return the reps as is, which would be 100 in the example, because the athlete finished the workout
    2. Scale up the number of reps to 100*20/15 = 133.33, which could approximate the number of reps the athlete would have finished in 20 minutes.
    """
    # TODO: Implement this function
    

def convert_time_cap_workout_to_time(x, total_reps, time_cap, scale_up=False):
    """
    Here x can either be in reps or a time. For example, if the workout had a total of 100 reps and a time cap of 20 minutes,
    and the athlete finished only finished 80 reps by the cap, we could do two things:
    1. Either return the time as is, which would be 20 in the example, because the athlete finished the workout
    2. Scale up the time to 20*100/80 = 25, which could approximate the time the athlete would have taken finished the workout. (Not necessarily though)
    """
    # TODO: Implement this function
    

def convert_to_floats(df, descriptions):
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

    for workout_name in df_modified.columns:
        # if we don't have the workout name in descriptions, we skip it
        if workout_name not in descriptions:
            continue

        # if workout is for REPS, we should be fine
        if descriptions[workout_name]["goal"].lower() == "reps":
            df_modified[workout_name] = df_modified[workout_name].astype(int)
            # TODO: handle any edge cases here

        elif descriptions[workout_name]["goal"].lower() == "for time":
            if descriptions[workout_name]["time_cap"] is not None:
                # if the workout has a time cap, some athletes might not have finished it.
                # The ones that finished would have a time, the rest would have reps completed until the timecap.
                df_modified[workout_name] = df_modified[workout_name].apply(
                    lambda x: convert_time_cap_workout_to_time(
                        x,
                        descriptions[workout_name]["reps"],
                        descriptions[workout_name]["time_cap"],
                    )
                )

            else:
                # if there is no time cap, we can just convert the time to a float
                df_modified[workout_name] = convert_to_datetime(df_modified[workout_name])

    return df_modified



def handle_outliers(df, score_headers = None):
    """
    This function will detect outliers in the data. and replace them with missing values
    """
    df_modified = df.copy()

    # If score_headers is None, use all columns
    if score_headers is None:
        score_headers = df.columns
        
    # Sanity check to confirm that the columns are numeric
    for col in score_headers:
        if df_modified[col].dtype not in [int, float]:
            raise ValueError(f"Column {col} is not numeric, convert to numeric first")
    

    # fill 0 with na first to prevent missing
    df_modified = df_modified.replace(0, np.nan)

    # find interquartile rang
    upper_quartiles = df_modified[score_headers].quantile(0.75)
    lower_quartiles = df_modified[score_headers].quantile(0.25) 
    iqr = upper_quartiles - lower_quartiles

    # find outliers
    outliers = (df_modified[score_headers] > (upper_quartiles + 1.5 * iqr)) | (df_modified[score_headers] < (lower_quartiles - 1.5 * iqr))
    df_modified[outliers] = np.nan

    return df_modified
