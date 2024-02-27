import pandas as pd
import numpy as np

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



def handle_outliers(df):
    """
    This function will detect outliers in the data. and replace them with missing values
    """
    # TODO: Implement this function
    df_modified = df.copy()
    column_names = df.columns.tolist()
    score_headers = []
    for column in column_names:
        if '_score' in column:
            score_headers.append(column) 
    for score in score_headers: 
        df_copy_score = df_modified[score]
        upperQuartile = df_copy_score.quantile(.75)
        lowerQuartile = df_copy_score.quantile(.25)
        iqr = upperQuartile - lowerQuartile
        for index, row in df_modified.iterrows():
            value = row[score]
            if ((value > (upperQuartile + 1.5*iqr)) | (value < (lowerQuartile - 1.5*iqr))).any(): 
                df_modified.loc[index, score] = '0'
    return df_modified

  
#df = pd.read_csv('Mens_Crossfit_data_cleaned.csv')
#print(handle_outliers(df))
