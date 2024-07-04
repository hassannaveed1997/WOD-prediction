import pandas as pd
from .base import BaseFEPipelineObject
from .helpers import convert_to_floats, seperate_scaled_workouts
from ..constants import Constants as c

class OpenResultsFE(BaseFEPipelineObject):
    """
    Feature engineering pipeline object for processing open results data.

    Args:
        create_description_embeddings (bool): Flag indicating whether to create description embeddings. Default is False.
        scale_up (bool): Flag indicating whether to scale up the data. Default is False.
        **kwargs: Additional keyword arguments.

    Attributes:
        create_description_embeddings (bool): Flag indicating whether to create description embeddings.
        scale_up (bool): Flag indicating whether to scale up the data.
        kwargs (dict): Additional keyword arguments.

    Methods:
        melt_data(open_data): Melt the data into a long format.
        description_embeddings(): Generate description embeddings.
        transform(open_data, workout_descriptions=None): Perform various operations on the data.

    """

    def __init__(
        self, create_description_embeddings=False, scale_up=False, **kwargs
    ):
        self.create_description_embeddings = create_description_embeddings
        self.scale_up = scale_up
        self.kwargs = kwargs

        super().__init__()

    def melt_data(self, open_data):
        """
        Melt the data into a long format.

        Args:
            open_data (pd.DataFrame): The input data to be melted.

        Returns:
            pd.DataFrame: The melted data.

        """
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
        """
        Generate description embeddings.

        Raises:
            NotImplementedError: This method is not implemented.

        """
        raise NotImplementedError

    def transform(self, open_data, workout_descriptions=None):
        """
        Perform various operations on the data.

        This function is intended to perform a few operations:
        - convert workout columns into a single data type
        - melt the data into a long format (if applicable). Then each row will represent a single workout, rather than a single athlete.
        - add relevant info from workout descriptions

        Args:
            open_data (pd.DataFrame): The input data to be transformed.
            workout_descriptions (pd.DataFrame, optional): The workout descriptions. Default is None.

        Returns:
            pd.DataFrame: The transformed data.

        Notes:

        """
        # separate out scaled workouts as separate columns
        open_data = seperate_scaled_workouts(open_data)

        # convert to floats (instead of reps, lbs time or mixed data types)
        open_data = convert_to_floats(
            open_data, workout_descriptions, scale_up=self.scale_up
        )

        open_data = self.melt_data(open_data)

        # store mapping from melted data
        self.create_meta_data(open_data)

        if self.create_description_embeddings:
            description_embeddings = self.description_embeddings(
                workout_descriptions
            )
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
            workout_name_mapping = "workout_"+ data[workout_cols[0]]
        else:
            # get argmax
            workout_name_mapping = data[workout_cols].idxmax(axis=1)
        return workout_name_mapping
