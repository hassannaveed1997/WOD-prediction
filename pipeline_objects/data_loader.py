import pandas as pd
import numpy as np
import os
import json


class DataLoader:
    """
    A class for loading data for WOD prediction.

    Loads the open results, athlete info (not implemented), workout
    descriptions, and benchmark stats data into a dictionary with keys:
        - open_results
        - athlete_info (not implemented)
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
        load: Loads the specified data objects.

    """

    def __init__(self, root_path, objects):
        # see if the root path exists
        self.root_path = root_path
        if not os.path.exists(root_path):
            raise FileNotFoundError("The root path does not exist")
        self.objects = objects

    def load_open_results(self):
        """
        Loads the open results data.

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
        Loads the athlete info data.

        Returns:
            None

        """
        # TODO: load the athlete info
        pass

    def load_descriptions(self):
        """
        Loads the workout descriptions data.

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
                "benchmark_stats/2023_BenchMarkStats_men3_cleaned.csv",
            )
        )
        if "Unnamed: 0" in benchmark_stats.columns:
            benchmark_stats.drop(columns=["Unnamed: 0"], inplace=True)

        benchmark_stats.set_index("id", inplace=True)

        return benchmark_stats

    def load(self):
        """
        Loads the specified data objects.

        Load the data objects specified in the objects attribute to a
        dictionary with keys:
            - open_results
            - athlete_info (not implemented)
            - workout_descriptions
            - benchmark_stats

        Returns:
            dict: A dictionary containing the loaded data objects.

        """
        data = {}
        if "open_results" in self.objects:
            open_results = self.load_open_results()
            data["open_results"] = open_results

        # TODO: add for other 3 input sources
        if "athlete_info" in self.objects:
            raise NotImplementedError(
                "The athlete info is not implemented yet"
            )

        if "descriptions" in self.objects:
            descriptions = self.load_descriptions()
            data["workout_descriptions"] = descriptions

        if "benchmark_stats" in self.objects:
            benchmark_stats = self.load_benchmark_stats()
            data["benchmark_stats"] = benchmark_stats

        return data
