import pandas as pd
import numpy as np
from .base import BaseFEPipelineObject
from .helpers import convert_to_floats, seperate_scaled_workouts
from ..constants import Constants as c
from .normalization import PercentileScaler, StandardScalerByWod

class OpenResultsFE(BaseFEPipelineObject):
    def __init__(self, create_description_embeddings=False, scale_up=False, scale_method = None, **kwargs):
        self.create_description_embeddings = create_description_embeddings
        self.scale_up = scale_up
        self.scale_method = scale_method
        self.kwargs = kwargs

        super().__init__()

    def melt_data(self, open_data):
        open_data_melted = open_data.melt(
            var_name="workout", value_name="score", ignore_index=False
        )
        # get rid of missing values
        open_data_melted = open_data_melted.dropna(subset=["score"])

        # recreate index
        open_data_melted[c.athlete_id_col] = open_data_melted.index
        open_data_melted.index = self.create_index(
            open_data_melted.index, open_data_melted["workout"]
        )
        self.meta_data['idx_to_athlete_id'] = open_data_melted[c.athlete_id_col]
        return open_data_melted
    
    @staticmethod
    def create_index(athlete_ids, workout_ids):
        """
        creates new index post melting with athlete id and workout id concatenated
        """
        #make the names smaller
        workout_ids = workout_ids.str.replace("_scaled", "s")
        workout_ids = workout_ids.str.replace("_foundation", "f")
        workout_ids = workout_ids.str.replace(".", "_")

        #concatenate the two
        index = athlete_ids.astype(str) + "_" + workout_ids
        return index


    def description_embeddings(self):
        raise NotImplementedError

    def transform(self, open_data, workout_descriptions=None):
        """
        This function is intended to perform a few operations:
        - convert workout columns into a single data type
        - melt the data into a long format (if applicable). Then each row will represent a single workout, rather than a single athlete.
        - add relevant info from workout descriptions
        """
        # seperate out scaled workouts as seperate columns
        open_data = seperate_scaled_workouts(open_data)

        # convert to floats (instead of reps, lbs time or mixed data types)
        open_data = convert_to_floats(
            open_data, workout_descriptions, scale_up=self.scale_up
        )

        # TODO: normalize data here
        open_data = self.scale_data(open_data)

        # convert to percentiles (if requested)
        open_data = self.melt_data(open_data)

        # store mapping from melted data
        self.create_meta_data(open_data)

        if self.create_description_embeddings:
            description_embeddings = self.description_embeddings(workout_descriptions)
            # merge the two datasets
            open_data = pd.merge(
                open_data, description_embeddings, how="left", on="workout"
            )

        return open_data
    
    def create_meta_data(self, df):
        self.meta_data['idx_to_workout_name'] = self.get_workout_name_mapping(df)
        return

    @staticmethod
    def get_workout_name_mapping(data):
        workout_cols = data.columns[data.columns.str.contains("workout")]

        if len(workout_cols) == 1:
            workout_name_mapping = c.workout_prefix + data[workout_cols[0]]
        else:
            # get argmax
            workout_name_mapping = data[workout_cols].idxmax(axis=1)
        return workout_name_mapping
    
    def scale_data(self, df):
        if self.scale_method is None:
            return df
        elif self.scale_method == 'percentile':
            scaler = PercentileScaler()
        elif self.scale_method == 'standard':
            scaler = StandardScalerByWod()
        else:
            raise ValueError('Invalid scaling method. Must be either percentile or standard.')

        df = scaler.transform(df)
        self.meta_data['scaler'] = scaler
        return df
