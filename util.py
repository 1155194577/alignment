from typing import List, Tuple, Union, Dict
import numpy as np
import pandas as pd

def has_negative_times(events: List[Dict]) -> bool:
    """
    Check if any event has negative time values.
    
    Args:
        events: List of event dictionaries with 'start_time' and 'end_time' keys
        
    Returns:
        True if any event has negative time values, False otherwise
    """
    return any(event['start_time'] < 0 or event['end_time'] < 0 for event in events)

def calculate_cosine_similarity(
    vector1: np.ndarray,
    vector2: np.ndarray
) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vector1: First input vector
        vector2: Second input vector
        
    Returns:
        Cosine similarity value between 0 and 1
    """
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def calculate_mean_squared_error(
    vector1: np.ndarray,
    vector2: np.ndarray
) -> float:
    """
    Calculate the mean squared error between two vectors.
    
    Args:
        vector1: First input vector
        vector2: Second input vector
        
    Returns:
        Mean squared error value
    """
    if len(vector1) != len(vector2):
        raise ValueError("Input vectors must have the same length")
    return np.mean((vector1 - vector2) ** 2)

def calculate_mean_absolute_error(
    vector1: np.ndarray,
    vector2: np.ndarray
) -> float:
    """
    Calculate the mean absolute error between two vectors.
    
    Args:
        vector1: First input vector
        vector2: Second input vector
        
    Returns:
        Mean absolute error value
    """
    if len(vector1) != len(vector2):
        raise ValueError("Input vectors must have the same length")
    return np.mean(np.abs(vector1 - vector2))

def binarize_array(
    array: np.ndarray,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Convert an array to binary values based on threshold.
    
    Args:
        array: Input array to binarize
        threshold: Value above which elements become 1, 0 otherwise
        
    Returns:
        Binary array with values 0 or 1
    """
    return (array > threshold).astype(int)

def calculate_dtw_distance(
    array1: np.ndarray,
    array2: np.ndarray
) -> float:
    """
    Calculate Dynamic Time Warping (DTW) distance between two arrays.
    
    Args:
        array1: First input array
        array2: Second input array
        
    Returns:
        DTW distance between the arrays
    """
    n, m = len(array1), len(array2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[0, 1:] = float('inf')
    dtw_matrix[1:, 0] = float('inf')
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(array1[i - 1] - array2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )
    
    return dtw_matrix[n, m]

def calculate_mean_difference(
    array1: np.ndarray,
    array2: np.ndarray
) -> float:
    """
    Calculate the absolute difference between means of two arrays.
    
    Args:
        array1: First input array
        array2: Second input array
        
    Returns:
        Absolute difference between means
    """
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    return abs(mean1 - mean2)

def compute_similarity_metric(
    reference: np.ndarray,
    target: np.ndarray,
    metric: str = 'cosine',
    binarize: bool = False
) -> float:
    """
    Compute similarity metric between two arrays.
    
    Args:
        reference: Reference array
        target: Target array to compare against reference
        metric: Similarity metric to use ('cosine', 'mse', 'dtw', 'mae', 'mean_difference')
        binarize: Whether to binarize the arrays before computing metric
        
    Returns:
        Similarity metric value
    """
    if binarize:
        reference = binarize_array(reference)
        target = binarize_array(target)
    
    if metric == 'cosine':
        return calculate_cosine_similarity(reference, target)
    elif metric == 'mse':
        return calculate_mean_squared_error(reference, target)
    elif metric == 'dtw':
        return calculate_dtw_distance(reference, target)
    elif metric == 'mae':
        return calculate_mean_absolute_error(reference, target)
    elif metric == 'mean_difference':
        return calculate_mean_difference(reference, target)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def convert_search_record_to_dataframe(
    search_record: List[Tuple]
) -> pd.DataFrame:
    """
    Convert search record to pandas DataFrame for analysis.
    
    Args:
        search_record: List of tuples containing search results
        
    Returns:
        DataFrame with search results
    """
    column_names = [
        'vertical_shift',
        'horizontal_shift',
        'scaling_factor',
        'cosine_similarity',
        'binary_cosine_similarity',
        'mse',
        'binary_mse',
        'mae',
        'binary_mae',
        'mean_difference'
    ]
    
    return pd.DataFrame(search_record, columns=column_names)
