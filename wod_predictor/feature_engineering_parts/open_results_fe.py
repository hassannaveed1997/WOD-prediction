import os

import pandas as pd

from wod_predictor.feature_engineering_parts.base import TransformerMixIn
from wod_predictor.helpers import get_base_path

from ..constants import Constants as c
from .utils.embeddings import LLMClient, reduce_dimensions_pca
from .utils.helpers import (convert_to_floats, remove_scaled_workout_columns,
                            remove_suffixes, seperate_scaled_workouts)
from .utils.normalization import (GenericSklearnScaler, QuantileScaler,
                                  StandardScalerByWod)


class OpenResultsFE(TransformerMixIn):
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
        self,
        create_description_embeddings=False,
        conversion_method="rpm",
        scale_up=False,
        scale_args=None,
        allow_modified=True,
        embedding_dim=100,
        **kwargs
    ):
        self.columns = []
        self.create_description_embeddings = create_description_embeddings
        self.conversion_method = conversion_method
        self.scale_up = scale_up
        self.allow_modified = allow_modified
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs
        self.scaler = self.initialize_scaler(scale_args)
        self.meta_data = {}

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
        open_data_melted[c.year_col] = open_data_melted["workout"].str.slice(0, 2)
        open_data_melted.index = self.create_index(
            open_data_melted.index, open_data_melted["workout"]
        )
        self.meta_data["idx_to_athlete_id"] = open_data_melted[c.athlete_id_col]
        return open_data_melted

    @staticmethod
    def create_index(athlete_ids, workout_ids):
        """
        creates new index post melting with athlete id and workout id concatenated
        """
        # make the names smaller
        workout_ids = workout_ids.str.replace(c.scaled_tag, "s")
        workout_ids = workout_ids.str.replace(c.foundation_tag, "f")
        workout_ids = workout_ids.str.replace(".", "_")

        # concatenate the two
        index = athlete_ids.astype(str) + "_" + workout_ids
        return index

    def description_embeddings(self, workout_descriptions):
        """
        Generate description embeddings.
        """
        if workout_descriptions is None:
            raise ValueError(
                "Workout descriptions must be provided to create description embeddings"
            )

        # read cache to get embeddings
        wod_prediction_path = get_base_path()
        embedding_cache_path = os.path.join(
            wod_prediction_path, "cache/workout_embeddings_cache.csv"
        )
        embeddings_df = pd.read_csv(embedding_cache_path)
        client = LLMClient()
        new_embeddings = {}
        for key, value in workout_descriptions.items():
            if key not in embeddings_df.columns:
                new_embeddings[key] = client.get_embedding(value["description"])

        # add new embeddings to the dataframe
        new_embeddings_df = pd.DataFrame(new_embeddings)
        embeddings_df = pd.concat([embeddings_df, new_embeddings_df], axis=1)

        # save the new embeddings to the cache
        embeddings_df.to_csv(embedding_cache_path, index=False)

        # reduce the dimensions of the embeddings
        if self.embedding_dim is not None:
            embeddings_df = reduce_dimensions_pca(
                embeddings_df, n_components=self.embedding_dim
            )

        # convert to format for use
        embeddings_df = embeddings_df.T
        embeddings_df.columns = [
            "embedding_dim_" + str(i) for i in range(embeddings_df.shape[1])
        ]
        embeddings_df.reset_index(names="workout", inplace=True)
        return embeddings_df

    def _preprocess(self, open_data, workout_descriptions):
        # get a list of all possible columns
        temp_df = seperate_scaled_workouts(open_data)
        if not self.allow_modified:
            temp_df = remove_scaled_workout_columns(temp_df)
        temp_df = remove_suffixes(temp_df)

        # convert to floats (instead of reps, lbs time or mixed data types)
        temp_df = convert_to_floats(
            temp_df,
            workout_descriptions,
            conversion_method=self.conversion_method,
            scale_up=self.scale_up,
        )
        return temp_df

    def fit(self, open_data):
        """
        Initialize any transformers for later use in transform
        """

        if not self.create_description_embeddings:
            self.columns += list(open_data.columns)
        super().fit()

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
        """
        open_data = self._preprocess(open_data, workout_descriptions)
        self.check_fit(open_data = open_data)

        # scale data if possible
        if self.scaler:
            open_data = self.scaler.transform(open_data)

        # convert to percentiles (if requested)
        open_data = self.melt_data(open_data)

        # store mapping from melted data
        self.create_meta_data(open_data)

        if self.create_description_embeddings:
            description_embeddings = self.description_embeddings(workout_descriptions)
            # merge the two datasets
            open_data_transformed = pd.merge(
                open_data, description_embeddings, how="left", on="workout"
            )
            open_data_transformed.index = open_data.index
        else:
            # one hot encode the workouts
            workout_dummies = pd.get_dummies(open_data["workout"]).astype(float)
            # ensure that the columns are in the same order
            for col in self.columns:
                if col not in workout_dummies.columns:
                    workout_dummies[col] = 0
            workout_dummies = workout_dummies[self.columns]
            workout_dummies.columns = [
                c.workout_col_prefix + col for col in workout_dummies.columns
            ]

            # add back
            open_data_transformed = pd.concat([open_data, workout_dummies], axis=1)

        open_data_transformed.drop(columns=["workout"], inplace=True)
        return open_data_transformed

    def create_meta_data(self, df):
        self.meta_data["idx_to_workout_name"] = self.get_workout_name_mapping(df)
        if self.scaler is not None:
            self.meta_data["scaler"] = self.scaler
        return

    @staticmethod
    def get_workout_name_mapping(data):
        workout_cols = data.columns[data.columns.str.contains("workout")]

        if len(workout_cols) == 1:
            workout_name_mapping = c.workout_col_prefix + data[workout_cols[0]]
        else:
            # get argmax
            workout_name_mapping = data[workout_cols].idxmax(axis=1)
        return workout_name_mapping

    def initialize_scaler(self, scale_args):
        if scale_args is None or "method" not in scale_args:
            return None

        scale_args_ = scale_args.copy()
        scale_method = scale_args_.pop("method")
        if scale_method == "quantile":
            scaler = QuantileScaler()
        elif scale_method == "standard":
            scaler = StandardScalerByWod()
        elif scale_method == "general":
            assert (
                "scaler_name" in scale_args
            ), "To use general scaler, you must specify scaler_name"
            scaler = GenericSklearnScaler(**scale_args_)
        else:
            raise ValueError(
                "Invalid scaling method. Must be either percentile, standard, or general."
            )
        return scaler
