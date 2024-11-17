import pandas as pd
from .base import BaseFEPipelineObject
from .helpers import fill_missing_values, convert_units
from ..constants import Constants as c
from datetime import datetime


class AthleteInfoFE(BaseFEPipelineObject):
    """
    Feature engineering pipeline object for athlete information.
    """

    def __init__(self, missing_method="zero", **kwargs):
        self.missing_method = missing_method
        self.kwargs = kwargs

    def fit(self, athlete_info_data: pd.DataFrame):
        """
        Add any initial setup here if needed.
        """
        pass

    def transform(self, athlete_info_data: pd.DataFrame):
        """
        Transforms athlete info data:
        - melts data from wide to long format, with each row representing an athlete/year pair
        - converts units to metric
        - creates features from athlete info data
        - calculates and verifies birth year consistency across all reported years
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

        # Calculate birth year based on each reported year and verify consistency
        athlete_info_numeric['birth_year_calculated'] = (
            athlete_info_numeric['year'].astype(int) - athlete_info_numeric['age']
        )
        
        # Check for consistency in calculated birth year
        birth_year_consistency = (
            athlete_info_numeric.groupby(c.athlete_id_col)['birth_year_calculated']
            .nunique()
            .reset_index(name='unique_birth_years')
        )
        inconsistent_athletes = birth_year_consistency[
            birth_year_consistency['unique_birth_years'] > 1
        ][c.athlete_id_col]

        # Flag or handle inconsistencies (e.g., log, remove, etc.)
        if not inconsistent_athletes.empty:
            # Optionally, log or handle inconsistent athletes
            print("Inconsistent birth year calculations for athletes:", inconsistent_athletes.tolist())
            
            # Remove inconsistent athletes (optional)
            athlete_info_numeric = athlete_info_numeric[
                ~athlete_info_numeric[c.athlete_id_col].isin(inconsistent_athletes)
            ]

        # Drop temporary birth_year_calculated column and keep verified birth_year
        athlete_info_numeric['birth_year'] = athlete_info_numeric.groupby(c.athlete_id_col)['birth_year_calculated'].transform('first')
        athlete_info_numeric.drop(columns=['birth_year_calculated'], inplace=True)

        # Create features from athlete info data
        athlete_info_with_features = self.create_features(athlete_info_numeric)

        # Fill missing values if specified
        if self.missing_method is not None:
            athlete_info_with_features = fill_missing_values(
                athlete_info_with_features, method=self.missing_method, **self.kwargs
            )
        return athlete_info_with_features

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
