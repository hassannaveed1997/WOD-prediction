import pandas as pd

def convert_to_floats(data):
    # TODO: This is just a placeholder to proceed. We need to migrate the function from modeling.preprocessing to here
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].str.replace(' reps','')
                data[col] = data[col].str.replace(' lbs','')
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except ValueError:
                raise ValueError(f"Could not convert column {col} to float")
    return data

class OpenResultsFE:
    def __init__(self,create_description_embeddings = False, **kwargs):
        self.create_description_embeddings = create_description_embeddings
        self.kwargs = kwargs

    def melt_data(self, open_data):
        open_data = open_data.melt(var_name = 'workout',value_name='score', ignore_index=False).reset_index()

        # get rid of missing values
        open_data = open_data.dropna(subset = ['score'])
        return open_data
    
    def description_embeddings(self):
        raise NotImplementedError

    def transform(self, open_data, workout_descriptions = None):
        """
        This function is intended to perform a few operations:
        - convert workout columns into a single data type
        - melt the data into a long format (if applicable). Then each row will represent a single workout, rather than a single athlete.
        - add relevant info from workout descriptions
        """
        open_data = convert_to_floats(open_data)

        open_data = self.melt_data(open_data)
        
        if self.create_description_embeddings:
            description_embeddings = self.description_embeddings(workout_descriptions)
            # merge the two datasets
            open_data = pd.merge(open_data, description_embeddings, how='left', on='workout')
        
        return open_data