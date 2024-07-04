import pandas as pd
from .base import BaseFEPipelineObject
from .helpers import fill_missing_values, remove_outliers


class AthleteInfoFE(BaseFEPipelineObject):
    """
    Feature engineering pipeline object for athlete information.

    TODO: IMPLEMENT THIS CLASS FULLY
    """

    def __init__(
        self, remove_outliers: bool = True, missing_method="knn", **kwargs
    ):
        # raise NotImplementedError("Athlete info FE class not yet implemented")
        self.remove_outliers = remove_outliers
        self.missing_method = missing_method
        self.kwargs = kwargs

    def transform(self, athlete_info_data: pd.DataFrame):
        """
        TODO: IMPLEMENT THIS METHOD FULLY

        TODO: Verify if remove_outliers and fill_missing_values will
        TODO: work for athlete info data.

        TODO:
            Need to modify missing data filling etc. to work with
            athlete info data.
        """

        # raise NotImplementedError(
        #     "Athlete info FE tranformation not yet implemented"
        # )

        # remove unnecessary columns
        if "name" in athlete_info_data.columns:
            athlete_info_data.drop(columns=["name"], inplace=True)

        # remove outliers
        if self.remove_outliers:
            athlete_info_data = remove_outliers(
                athlete_info_data, **self.kwargs
            )

        # fill missing values
        if self.missing_method is not None:
            athlete_info_data = fill_missing_values(
                athlete_info_data, method=self.missing_method, **self.kwargs
            )

        # Before resetting index (currently set to 'id' AKA athlete ID),
        # create a new column to save/store the athlete ID (to be used
        # later for reference/merging with other dataframes).
        # Add it as the first column in the dataframe.
        athlete_info_data.insert(0, "athlete_id", athlete_info_data.index)

        # reset index (better for merging downstream)
        athlete_info_data = athlete_info_data.reset_index()
