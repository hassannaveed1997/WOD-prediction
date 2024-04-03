
from sklearn.decomposition import PCA
from openai import OpenAI
import pandas as pd

def get_embedding(text, model="text-embedding-3-small"):
   client = OpenAI()

   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def reduce_dimensions_pca(embeddings, n_components=2):
    """
    Uses PCA to reduce the dimensions of the embeddings to 2D
    Parameters:
    ----------
    embeddings: pd.DataFrame
        The embedding to reduce the dimensions of
    
    Returns:
    -------
    embeddings_2d_pca: np.array
        The 2D embeddings
    """
    pca = PCA(n_components=n_components)
    embeddings_reduced_pca = pca.fit_transform(embeddings.T)
    # convert back to df
    embeddings_reduced_pca = pd.DataFrame(embeddings_reduced_pca.T, columns=embeddings.columns)
    return embeddings_reduced_pca