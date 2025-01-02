import pandas as pd

from wod_predictor.feature_engineering_parts.base import TransformerMixIn
from wod_predictor.feature_engineering_parts.utils.missing_values import \
    MissingValueImputation

from ..constants import Constants as c
from .utils.helpers import convert_units


class AthleteInfoFE(TransformerMixIn):
    """
    Feature engineering pipeline object for athlete information.
    """

    def __init__(self, missing_method="zero", **kwargs):
        # Fill missing values if specified
        self.transformers = []
        if missing_method is not None:
            self.transformers.append(MissingValueImputation(method=missing_method, **kwargs))
        super().__init__()

    def transform(self, athlete_info_data: pd.DataFrame):
        """
        Transforms athlete info data:
        - melts data from wide to long format, with each row representing an athlete/year pair
        - converts units to metric
        - fills missing values if specified

        Parameters:
            athlete_info_data (pd.DataFrame): Athlete info data

        Returns:
            pd.DataFrame: Transformed athlete info data
        """
        # Melt data into long format
        athlete_info_melted = self.melt(athlete_info_data)

        # Convert data types to numeric
        athlete_info_numeric = self.fix_units(athlete_info_melted)

        # run any available transformers
        for transformer in self.transformers:
            athlete_info_numeric = transformer.transform(athlete_info_numeric)

        return athlete_info_numeric.drop(columns=["name"])

    def melt(self, athlete_info_data: pd.DataFrame):
        """
        Flattens the athlete info data to long format with year information.
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
        # if the name is missing, they didn't participate in the open that year
        melted_athlete_info.dropna(subset=["name"], inplace=True)
        return melted_athlete_info

    def fix_units(self, athlete_info_data: pd.DataFrame):
        """
        Convert units of athlete info data.
        """
        athlete_info_data["year"].astype(int)

        athlete_info_data["height"] = convert_units(
            athlete_info_data[["height"]], type="height"
        )
        athlete_info_data["weight"] = convert_units(
            athlete_info_data[["weight"]], type="weight"
        )

        return athlete_info_data
