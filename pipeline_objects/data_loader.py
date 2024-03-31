import pandas as pd
import numpy as np
import os

class DataLoader():
    def __init__(self, root_path, objects):
        # see if the root path exists
        self.root_path = root_path
        if not os.path.exists(root_path):
            raise FileNotFoundError("The root path does not exist")
        self.objects = objects
    

    def load_open_results(self):
        # load the open results
        files= os.listdir(self.root_path)
        open_results = []
        for file in files:
            if file.endswith('scores.csv'):
                # year = file.split('_')[0]
                df = pd.read_csv(os.path.join(self.root_path, file))

                # remove the columns that are not needed
                df.set_index('id', inplace = True)
                if 'Unnamed: 0' in df.columns:
                    df.drop(columns = ['Unnamed: 0'], inplace = True)

                open_results.append(df)

        # concatenate the results
        open_results = pd.concat(open_results, axis = 1)
        return open_results
    

    def load_athlete_info(self):
        # TODO: load the athlete info
        pass

    def load_descriptions(self):
        # TODO: load the description
        pass

    def load_benchmark_stats(self):
        # TODO: load the benchmark stats
        pass
        
    def load(self):
        data= {}
        if 'open_results' in self.objects:
            open_results = self.load_open_results()
            data['open_results'] = open_results
        
        # TODO: add for other 3 input sources
        if 'athlete_info' in self.objects:
            raise NotImplementedError("The athlete info is not implemented yet")
        
        if 'descriptions' in self.objects:
            raise NotImplementedError("The description is not implemented yet")
        
        if 'benchmark_stats' in self.objects:
            raise NotImplementedError("The benchmark stats is not implemented yet")
            
        return data
    