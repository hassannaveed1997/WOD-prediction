from pipeline_objects.feature_engineering_parts.open_results_fe import OpenResultsFE
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
            raise NotImplementedError("Benchmark stats transformation not yet implemented")
        
        if 'athlete_info' in self.config:
            raise NotImplementedError("Athlete info transformation not yet implemented")

        
        fe_data = pd.concat(fe_data, axis = 1)  
        output = {
            'X': fe_data,
            'y': y
        }   
        return output
        