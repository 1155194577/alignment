from math import sqrt
import numpy as np
import pandas as pd
def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two numpy arrays."""
    num = np.dot(vec1, vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return num / denom if denom != 0 else 0.0

def mean_squared_error(vec1, vec2):
    """Calculate the mean squared error between two numpy arrays."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")
    return np.mean((vec1 - vec2) ** 2)

def binarise_vector(vec, threshold=0.0):
    """
    Convert a numpy array to a binary array based on a threshold.
    Elements greater than the threshold become 1, else 0.
    """
    return (vec > threshold).astype(int) 

def dtw_distance(vec1, vec2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two vectors.
    This is a simplified version and may not handle all edge cases.
    """
    n, m = len(vec1), len(vec2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = float('inf')
    dtw_matrix[1:, 0] = float('inf')
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(vec1[i - 1] - vec2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],      # insertion
                                           dtw_matrix[i, j - 1],      # deletion
                                           dtw_matrix[i - 1, j - 1])  # match

    return dtw_matrix[n, m]

def compute_similarity_metric(reference_contour, transformed_contour, metric='cosine', binarise=False):
    """
    Compute similarity metric between two contours.
    Supported metrics: 'cosine', 'mse'
    Assumes inputs are numpy arrays.
    Optionally binarises the contours before computing the metric.
    """
    if binarise:
        reference_contour = binarise_vector(reference_contour)
        transformed_contour = binarise_vector(transformed_contour)
    if metric == 'cosine':
        return cosine_similarity(reference_contour, transformed_contour)
    elif metric == 'negative-mse':
        return -mean_squared_error(reference_contour, transformed_contour)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def convert_search_record_to_dataframe(search_record): 
    """
    Convert the search record to a pandas DataFrame for easier analysis and visualization.
    Parameters:
    search_record (list of tuples): Each tuple contains (vertical_displacement, horizontal_displacement, scaling_factor, similarity).
    Returns:
    pd.DataFrame: A DataFrame with columns for vertical displacement, horizontal displacement, scaling factor, and similarity.
    """
    df = pd.DataFrame(search_record, columns=['vertical_displacement', 'horizontal_displacement', 'scaling_factor', 'similarity', 'negative_mse'])
    return df
