import pandas as pd
from .base import BaseFEPipelineObject
from .helpers import fill_missing_values, convert_units
from ..constants import Constants as c


class AthleteInfoFE(BaseFEPipelineObject):
    """
    Feature engineering pipeline object for athlete information.
    """

    def __init__(self, missing_method="zero", **kwargs):
        self.missing_method = missing_method
        self.kwargs = kwargs

    def fit(self, athlete_info_data: pd.DataFrame):
        """
        Add any initial setup here if needed
        """
        pass

    def transform(self, athlete_info_data: pd.DataFrame):
        """
        Transforms athlete info data
        - melts data from wide to long format, with each row representing an athlete/year pair
        - converts units to metric
        - creates features from athlete info data
        - OPTIONAL: fills missing values

        Parameters:
            athlete_info_data (pd.DataFrame): Athlete info data

        Returns:
            pd.DataFrame: Transformed athlete info data
        """
        athlete_info_melted = self.melt(athlete_info_data)

        # Tconvert data types to numeric
        athlete_info_numeric = self.fix_units(athlete_info_melted)

        # create features from athlete info data
        athlete_info_with_features = self.create_features(athlete_info_numeric)

        # fill missing values
        if self.missing_method is not None:
            athlete_info_with_features = fill_missing_values(
                athlete_info_with_features, method=self.missing_method, **self.kwargs
            )
        return athlete_info_with_features

    def melt(self, athlete_info_data: pd.DataFrame):
        """
        Flattens the athlete info data. We have multiple
        """
        years = set(athlete_info_data.columns.str.slice(0, 2))
        data_by_year = []
        for year in years:
            year_data = athlete_info_data.filter(like=year).copy()
            year_data.columns = year_data.columns.str.slice(3)
            year_data["year"] = year
            data_by_year.append(year_data)
        melted_athlete_info = pd.concat(data_by_year)

        athlete_ids = melted_athlete_info.index
        melted_athlete_info[c.athlete_id_col] = athlete_ids
        melted_athlete_info.index = (
            athlete_ids.astype(str) + "_" + melted_athlete_info["year"].astype(str)
        )

        return melted_athlete_info

    def create_features(self, athlete_info_data: pd.DataFrame):
        """
        Create new features from athlete info data.
        TODO: engineer any features here instead of just keeping most recent data.
        """
        athlete_info_data.drop_duplicates(c.athlete_id_col, keep="last", inplace=True)
        return athlete_info_data.drop(columns=["name", "age", "year"])

    def fix_units(self, athlete_info_data: pd.DataFrame):
        """
        Convert units of athlete info data.
        """
        athlete_info_data["height"] = convert_units(
            athlete_info_data[["height"]], type="height"
        )
        athlete_info_data["weight"] = convert_units(
            athlete_info_data[["weight"]], type="weight"
        )

        return athlete_info_data
