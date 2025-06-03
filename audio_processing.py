import pandas as pd
import numpy as np

def load_reference_contour(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the reference pitch contour and corresponding time values from a CSV file.

    Args:
        file_path (str): Path to the CSV file. The file must contain 'time' and 'value' columns.

    Returns:
        tuple[np.ndarray, np.ndarray]: (pitches, times)
    """
    df = pd.read_csv(file_path, usecols=['time', 'value'])
    times = df['time'].to_numpy(dtype=np.float64)  # Use seconds as is
    pitches = df['value'].to_numpy(dtype=np.float64)
    return pitches, times

