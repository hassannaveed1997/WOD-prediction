import pandas as pd
from .base import BaseFEPipelineObject
from .helpers import convert_to_floats, seperate_scaled_workouts


class OpenResultsFE(BaseFEPipelineObject):
    def __init__(self,create_description_embeddings = False, scale_up = False, **kwargs):
        self.create_description_embeddings = create_description_embeddings
        self.scale_up = scale_up
        self.kwargs = kwargs

    def melt_data(self, open_data):
        open_data = open_data.melt(var_name = 'workout',value_name='score', ignore_index=False).reset_index()

        # get rid of missing values
        open_data = open_data.dropna(subset = ['score'])
        return open_data
    
    def description_embeddings(self):
        raise NotImplementedError

    def transform(self, open_data, workout_descriptions = None):
        """
        This function is intended to perform a few operations:
        - convert workout columns into a single data type
        - melt the data into a long format (if applicable). Then each row will represent a single workout, rather than a single athlete.
        - add relevant info from workout descriptions
        """
        # seperate out scaled workouts as seperate columns
        open_data = seperate_scaled_workouts(open_data)

        # convert to floats (instead of reps, lbs time or mixed data types)
        open_data = convert_to_floats(open_data, workout_descriptions, scale_up = self.scale_up)

        open_data = self.melt_data(open_data)
        
        if self.create_description_embeddings:
            description_embeddings = self.description_embeddings(workout_descriptions)
            # merge the two datasets
            open_data = pd.merge(open_data, description_embeddings, how='left', on='workout')
        
        return open_data