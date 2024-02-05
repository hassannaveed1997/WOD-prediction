import pandas as pd
import numpy as np

def convert_to_floats(df, descriptions):
    """
    This function will take a workout from the open, such as 17.1, 17.3, etc. and convert it to a float. 
    TODO: Create descriptions of workouts from the open.
    Parameters:
    ----------
    df: pd.DataFrame
        The dataframe with the scores for the workout
    description: str
        The description of crossfit open workouts as a dictionary.
    Returns:
    -------
    df_modified: pd.DataFrame
        The modified dataframe with the scores for the workout as integers. This would either be reps, or time in minutes.
        These can be added as new columns. For example, 17.1 would become 17.1_reps if it was a workout for reps.
    """
    df_modified = df.copy()
    # INSERT CODE HERE
    return df_modified