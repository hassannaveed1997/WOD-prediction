import pandas as pd
from .helpers import get_embedding, reduce_dimensions_pca
import os

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
    def __init__(self,create_description_embeddings = False, embedding_dim = 100, **kwargs):
        self.create_description_embeddings = create_description_embeddings
        self.embedding_dim = embedding_dim
        self.kwargs = kwargs

    def melt_data(self, open_data):
        open_data = open_data.melt(var_name = 'workout',value_name='score', ignore_index=False).reset_index()

        # get rid of missing values
        open_data = open_data.dropna(subset = ['score'])
        return open_data
    
    def description_embeddings(self, workout_descriptions):
        if workout_descriptions is None:
            raise ValueError('Workout descriptions must be provided to create description embeddings')
        # get dir of this file
        

        # read cache to get embeddings
        embedding_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache/workout_embeddings_cache.csv')
        embeddings_df = pd.read_csv(embedding_cache_path)
        new_embeddings = {}
        for key, value in workout_descriptions.items():
            if key not in embeddings_df.columns:
                new_embeddings[key] = get_embedding(value['description'])
        
        # add new embeddings to the dataframe
        new_embeddings_df = pd.DataFrame(new_embeddings)
        embeddings_df = pd.concat([embeddings_df, new_embeddings_df], axis=1)
                
        # save the new embeddings to the cache
        embeddings_df.to_csv(embedding_cache_path, index=False)

        # reduce the dimensions of the embeddings
        if self.embedding_dim is not None:
            embeddings_df = reduce_dimensions_pca(embeddings_df, n_components=self.embedding_dim)
        
        # convert to format for use
        embeddings_df = embeddings_df.T
        embeddings_df.columns = ["emebedding_dim_" + str(i) for i in range(embeddings_df.shape[1])]
        return embeddings_df


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
            open_data = pd.merge(open_data, description_embeddings, how='left', left_on='workout', right_index=True, suffixes=("", "_embedding_dim"))
        
        return open_data