from sklearn.decomposition import PCA
from openai import OpenAI
import os
import pandas as pd
from wod_predictor.helpers import get_base_path
from wod_predictor.constants import OPEN_AI_API_KEY_PATH
import warnings


class LLMClient:
    def __init__(self):
        self.set_api_key()
        self.client = OpenAI()

    def set_api_key(self):
        if "OPENAI_API_KEY" not in os.environ:
            # set using default file
            api_key_path = get_base_path(levels_up=2) + OPEN_AI_API_KEY_PATH
            if not os.path.exists(api_key_path):
                raise FileNotFoundError(
                    f"Could not Open AI API file in f{api_key_path}. Please create file and save the key"
                )
            with open(api_key_path, "r") as reader:
                api_key = reader.read()
            os.environ["OPENAI_API_KEY"] = api_key
            print("API key set from file")

    def get_embedding(self, text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=model).data[0].embedding
        )


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
    min_eligeable_components = min(embeddings.shape)
    if n_components > min_eligeable_components:
        warnings.warn(
            f"PCA components cannot be greater than min(samples, features). Clipping from {n_components} to {min_eligeable_components}"
        )
        n_components = min_eligeable_components
    pca = PCA(n_components=n_components)
    embeddings_reduced_pca = pca.fit_transform(embeddings.T)
    # convert back to df
    embeddings_reduced_pca = pd.DataFrame(
        embeddings_reduced_pca.T, columns=embeddings.columns
    )
    return embeddings_reduced_pca
