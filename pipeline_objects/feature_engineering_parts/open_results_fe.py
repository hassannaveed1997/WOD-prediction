import pandas as pd

def convert_to_floats(data):
    # TODO: migrate the function from modeling.preprocessing to here
    return data

class OpenResultsFE:
    def __init__(self, melt = False, create_description_embeddings = False, **kwargs):
        self.melt = melt
        self.create_description_embeddings = create_description_embeddings
        self.kwargs = kwargs

    def melt_data(self):
        raise NotImplementedError
    
    def create_description_embeddings(self):
        raise NotImplementedError

    def transform(self, open_data, workout_descriptions = None):
        """
        This function is intended to perform a few operations:
        - convert workout columns into a single data type
        - melt the data into a long format (if applicable). Then each row will represent a single workout, rather than a single athlete.
        - add relevant info from workout descriptions
        """
        open_data = convert_to_floats(open_data)

        if self.melt:
            open_data = self.melt_data(open_data)
        
        if self.create_description_embeddings:
            description_embeddings = self.create_description_embeddings(workout_descriptions)
            # merge the two datasets
            open_data = pd.merge(open_data, description_embeddings, how='left', on='workout_id')
        
        return open_data