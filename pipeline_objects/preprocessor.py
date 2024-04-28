from pipeline_objects.feature_engineering_parts import OpenResultsFE,BenchmarkStatsFE
from functools import reduce
import pandas as pd

class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def transform(self, data):
        fe_data = []
        if 'open_results' in self.config:
            if 'open_results' not in data or 'workout_descriptions' not in data:
                raise ValueError('Both open results and workout descriptions must be provided to transform open results')
            
            open_results_fe = OpenResultsFE(**self.config['open_results'])
            open_results = open_results_fe.transform(data['open_results'], data.get('workout_descriptions', None))
            y = open_results['score']
            X = open_results.drop(columns = ['score'])

            fe_data.append(X)
        else:
            raise ValueError('Open results are needed to get scores')
            

        if 'benchmark_stats' in self.config:
            benchmark_stats_fe = BenchmarkStatsFE(**self.config['benchmark_stats'])
            benchmark_stats_transformed = benchmark_stats_fe.transform(data['benchmark_stats'])
            fe_data.append(benchmark_stats_transformed)
        
        if 'athlete_info' in self.config:
            raise NotImplementedError("Athlete info transformation not yet implemented")

        # join all feature engineered data together
        fe_data = reduce(lambda left, right: pd.merge(left, right, how = 'left'), fe_data)
        
        # one hot encode categorical variables
        for col in fe_data.columns:
            if fe_data[col].dtype == 'object':
                fe_data = pd.concat([fe_data, pd.get_dummies(fe_data[col], prefix=col)], axis=1)
                fe_data.drop(col, axis=1, inplace=True)
        
        
        output = {
            'X': fe_data,
            'y': y
        }   
        return output
        