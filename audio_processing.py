from typing import Tuple
import pandas as pd
import numpy as np

def load_reference_contour(
    file_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference pitch contour data from CSV file.
    
    Args:
        file_path: Path to CSV file containing time and value columns
        
    Returns:
        Tuple of (pitch_values, time_stamps) as numpy arrays
    """
    df = pd.read_csv(file_path, usecols=['time', 'value'])
    time_stamps = df['time'].to_numpy(dtype=np.float64)  # Time values in seconds
    pitch_values = df['value'].to_numpy(dtype=np.float64)  # Pitch values
    return pitch_values, time_stamps


def trim_contour_data(
    time_stamps: np.ndarray,
    pitch_values: np.ndarray,
    duration_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trim pitch contour data to specified duration.
    
    Args:
        time_stamps: Array of time values (ascending)
        pitch_values: Array of pitch values
        duration_threshold: Maximum duration to keep
        
    Returns:
        Tuple of (trimmed_pitch_values, trimmed_time_stamps)
    """
    mask = time_stamps < duration_threshold
    return pitch_values[mask], time_stamps[mask]