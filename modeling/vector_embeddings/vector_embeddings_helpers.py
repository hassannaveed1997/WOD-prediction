import json
import difflib
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from openai import OpenAI

client = OpenAI()

with open("../../Data/assets/movement_descriptions.json", "r") as file:
    movements = json.load(file)

def get_movement_description(movement):
    if movement not in movements:
        # find most similar movement
        closest_match = difflib.get_close_matches(movement, movements.keys(), n=1, cutoff=0.5)
        if closest_match:
            return movements[closest_match[0]]
        else:
            raise ValueError(f"Movement {movement} not found in the movement_descriptions.json")

    return movements[movement]

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def reduce_dimensions_tsne(embeddings):
    """
    The function will take embeddings and reduce the dimensions to 2D using TSNE
    
    Parameters:
    ----------
    embeddings: pd.DataFrame
        The embeddings to reduce the dimensions of. 

    Returns:
    -------
    embeddings_2d: np.array
        The 2D embeddings
    """
    embeddings_T = embeddings.T

    tsne = TSNE(n_components=2, random_state=0, perplexity=5)

    embeddings_2d = tsne.fit_transform(embeddings_T)

    return embeddings_2d

def visualize_embeddings_tsne(embeddings):
    """
    The function will take embeddings and visualize them in 2D using TSNE

    Parameters:
    ----------
    embeddings: pd.DataFrame
        The embeddings to visualize. Each column will be a text item's embeddings. 
        The number of rows corresponds to the number of dimensions in the embeddings.

    Returns:
    -------
    None
    """
    # Reduce the dimensions of the embeddings to 2D
    if embeddings.shape[0] > 2:
        embeddings_2d = reduce_dimensions_tsne(embeddings)
    else:
        embeddings_2d = embeddings.T

    # Create a DataFrame for the 2D embeddings
    df_2d = pd.DataFrame(embeddings_2d, columns=['tsne_1', 'tsne_2'])

    # Add passage names to the DataFrame
    df_2d['labels'] = embeddings.columns 

    # Plot using tsne that shows the name when you hover over
    fig = px.scatter(df_2d, x='tsne_1', y='tsne_2', text='labels')
    fig.update_traces(textposition='top center')
    fig.update_layout(title_text='TSNE')
    fig.show()

def reduce_dimensions_pca(embeddings):
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
    pca = PCA(n_components=2)
    embeddings_2d_pca = pca.fit_transform(embeddings.T)
    return embeddings_2d_pca

def visualize_embeddings_pca(embeddings):
    """
    The function will take embeddings and visualize them in 2D using PCA

    Parameters:
    ----------
    embeddings: pd.DataFrame
        The embeddings to visualize. Each column will be a text item's embeddings. 
        The number of rows corresponds to the number of dimensions in the embeddings.

    Returns:
    -------
    None
    """
    # visualize using pca
    if embeddings.shape[0] > 2:
        embeddings_2d_pca = reduce_dimensions_pca(embeddings)
    else:
        embeddings_2d_pca = embeddings.T

    df_2d_pca = pd.DataFrame(embeddings_2d_pca, columns=['pca_1', 'pca_2'])
    df_2d_pca['labels'] = embeddings.columns
    fig = px.scatter(df_2d_pca, x='pca_1', y='pca_2', text='labels')
    fig.update_traces(textposition='top center')
    fig.update_layout(title_text='PCA of Movement Descriptions')
    fig.show()

def visualize_embeddings(embeddings, method = 'tsne'):
    """
    Creates a 2D visualization of the embeddings using the specified method
    """
    if method == 'tsne':
        visualize_embeddings_tsne(embeddings)
    elif method == 'pca':
        visualize_embeddings_pca(embeddings)
    else:
        raise ValueError('method must be "tsne" or "pca"')
