import pandas as pd
import numpy as np
import re
EXCLUDED = ["workout_descriptions"]

class DataSpitter:
    def __init__(self, sample=None, test_ratio=0.2, test_filter=None) -> None:
        self.sample = sample
        self.test_ratio = test_ratio
        self.test_filter = test_filter

    def split(self, data):
        idx = data["open_results"].index
        # find intersection between all indices
        for key in data.keys():
            if key in EXCLUDED:
                continue
            idx = idx.intersection(data[key].index)

        if self.sample is not None:
            idx = np.random.choice(idx, self.sample, replace=False)

        test_idx = np.random.choice(idx, int(len(idx) * self.test_ratio), replace=False)
        train_idx = np.setdiff1d(idx, test_idx) 

        train_data = self.split_on_idx(data, train_idx)
        test_data = self.split_on_idx(data, test_idx)

        if self.test_filter is not None:
            filtered_cols = [
                col
                for col in test_data["open_results"].columns
                if re.search(self.test_filter, col)
            ]
            test_data["open_results"] = test_data["open_results"][filtered_cols]

        return train_data, test_data

    def split_on_idx(self, data, index):
        splitted_data = {}

        for key in data.keys():
            if key in EXCLUDED:
                splitted_data[key] = data[key]
            else:
                splitted_data[key] = data[key].loc[index]
            
        return splitted_data
