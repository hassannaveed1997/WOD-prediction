import json
import os

import numpy as np
import pandas as pd


class DataLoader:
    """
    A class for loading data for WOD prediction.

    Loads the open results, athlete info (age pending), workout
    descriptions, and benchmark stats data into a dictionary with keys:
        - open_results
        - athlete_info (age pending)
        - workout_descriptions
        - benchmark_stats

    Args:
        root_path (str): The root path where the data is located.
        objects (list): A list of objects to load.

    Attributes:
        root_path (str): The root path where the data is located.
        objects (list): A list of objects to load.

    Methods:
        load_open_results: Loads the open results data.
        load_athlete_info: Loads the athlete info data.
        load_descriptions: Loads the workout descriptions data.
        load_benchmark_stats: Loads the benchmark stats data.
        load: Loads ALL of the specified data objects.

    """

    def __init__(self, root_path, objects):
        # see if the root path exists
        self.root_path = root_path
        if not os.path.exists(root_path):
            raise FileNotFoundError("The root path does not exist")
        self.objects = objects

    def load_open_results(self):
        """
        Loads the open results data into a pandas DataFrame.

        Returns:
            pandas.DataFrame: The loaded open results data.

        """
        files = os.listdir(self.root_path)
        open_results = {}
        for file in files:
            if file.endswith("scores.csv"):
                year = file.split("_")[0]
                df = pd.read_csv(os.path.join(self.root_path, file))

                # remove the columns that are not needed
                df.set_index("id", inplace=True)
                if "Unnamed: 0" in df.columns:
                    df.drop(columns=["Unnamed: 0"], inplace=True)

                if year not in open_results:
                    open_results[year] = []
                open_results[year].append(df)
        # for each year, we need to concatenate the results along axis 0
        # (horizontally)
        for year in open_results:
            open_results[year] = pd.concat(open_results[year], axis=0)

        # concatenate the results (vertically)
        open_results = pd.concat(open_results.values(), axis=1)
        return open_results

    def load_athlete_info(self):
        """
        Loads the athlete info data for each year (not implemented).

        Returns:
            None

            .. code-block:: python
                # Appended to the end of the method before return
                sorted_cols = sorted(
                    [
                        col for col in athlete_info.columns
                        if col != 'id'
                    ],
                    key=lambda x: x.split('.')[0],
                    reverse=True,
                )
                if 'id' in athlete_info.columns:
                    sorted_cols = ['id'] + sorted_cols
                athlete_info = athlete_info[sorted_cols]

        """

        files = os.listdir(self.root_path)
        athlete_info = {}
        for file in files:
            if file.endswith("info.csv"):

                # Parse the file name for year and gender
                file_name_split = file.split("_")
                year = file_name_split[0]
                gender = file_name_split[1]

                # Create the DataFrame from the file content
                df = pd.read_csv(os.path.join(self.root_path, file))

                if gender == "Mens":
                    df["gender_male"] = 1
                else:
                    df["gender_male"] = 0

                df.set_index("id", inplace=True)
                # Drop the 'Unnamed: 0' column-which is just an index
                # in the original data without a column name
                if "Unnamed: 0" in df.columns:
                    df.drop(columns=["Unnamed: 0"], inplace=True)

                # Prepend the year to each column name, except 'id'
                df.columns = [
                    f"{year[-2:]}.{col}" if col != "id" else col for col in df.columns
                ]
                if year not in athlete_info:
                    athlete_info[year] = []
                athlete_info[year].append(df)

        # For each year, concatenate the results along axis 0
        # (vertically) to get the full dataset for that year
        #  This is for potential multiple files that have the same year
        for year in athlete_info:
            athlete_info[year] = pd.concat(athlete_info[year], axis=0)

        # Concatenate the results (horizontally) to get the full dataset
        # Essentially,
        athlete_info = pd.concat(athlete_info.values(), axis=1)

        return athlete_info

    def load_descriptions(self):
        """
        Loads the workout descriptions data into a dictionary.

        Returns:
            dict: The loaded workout descriptions data.

        """
        with open(
            os.path.join(
                self.root_path,
                "workout_descriptions/open_parsed_descriptions.json",
            ),
            "r",
        ) as f:
            descriptions = json.load(f)
        return descriptions

    def load_benchmark_stats(self):
        """
        Loads the benchmark stats data.

        Returns:
            pandas.DataFrame: The loaded benchmark stats data.

        """
        benchmark_stats = pd.read_csv(
            os.path.join(
                self.root_path,
                "benchmark_stats/Benchmark_stats_cleaned.csv",
            )
        )

        benchmark_stats.set_index("athlete_id", inplace=True)

        return benchmark_stats

    def load(self):
        """
        Load the data objects specified in the objects attribute to a
        dictionary with keys:
            - open_results (pandas.DataFrame)
            - athlete_info (not implemented)
            - workout_descriptions (dict)
            - benchmark_stats (pandas.DataFrame)

        Returns:
            dict: A dictionary containing the loaded data objects.
        """
        data = {}
        if "open_results" in self.objects:
            open_results = self.load_open_results()
            data["open_results"] = open_results

        if "athlete_info" in self.objects:
            athlete_info = self.load_athlete_info()
            data["athlete_info"] = athlete_info

        if "descriptions" in self.objects:
            descriptions = self.load_descriptions()
            data["workout_descriptions"] = descriptions

        if "benchmark_stats" in self.objects:
            benchmark_stats = self.load_benchmark_stats()
            data["benchmark_stats"] = benchmark_stats

        return data
