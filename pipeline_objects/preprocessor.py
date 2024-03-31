from pipeline_objects.feature_engineering_parts.open_results_fe import OpenResultsFE


class DataPreprocessor:
    def __init__(self, config):
        self.config = config

    def transform(self, data):
        transformed_data = {}
        if 'open_results' in self.config:
            if 'open_results' not in data or 'workout_descriptions' not in data:
                raise ValueError('Both open results and workout descriptions must be provided to transform open results')
            
            open_results_fe = OpenResultsFE(**self.config['open_results'])
            transformed_data['open_results'] = open_results_fe.transform(data['open_results'], data.get('workout_descriptions', None))

        if 'benchmark_stats' in self.config:
            bench
        
        return data
        