from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import mido
from tqdm import tqdm
from midi_processing import (
    transform_note_intervals,
    create_midi_function,
    create_midi_contour,
    convert_midi_to_pitch
)
from util import compute_similarity_metric, has_negative_times

def brute_force_search(
    reference_contour: np.ndarray,
    time_stamps: np.ndarray,
    note_intervals: List[Dict],
    search_space: Dict[str, np.ndarray],
    search_record: List[Tuple],
    metric: str = 'cosine',
    binarize: bool = False,
    maximize: bool = True,
    trim_threshold: float = 1.0
) -> Tuple[Tuple[float, float, float], float]:
    """
    Perform brute force search to find optimal alignment parameters.
    
    Args:
        reference_contour: Array of reference pitch values
        time_stamps: Array of time stamps
        note_intervals: List of MIDI note event intervals
        search_space: Dictionary containing parameter ranges
        search_record: List to store search results
        metric: Similarity metric to optimize
        binarize: Whether to binarize the data
        maximize: Whether to maximize the similarity metric
        trim_threshold: Fraction of total duration used for trimming (0-1)
        
    Returns:
        Tuple of (best_parameters, best_similarity)
    """
    best_similarity = float('-inf') if maximize else float('inf')
    best_parameters = None
    total_searches = (
        len(search_space['vertical']) * 
        len(search_space['horizontal']) * 
        len(search_space['scaling_factor'])
    )
    
    progress_bar = tqdm(total=total_searches, desc="Brute Force Search")
    
    for vertical_shift in search_space['vertical']:
        for horizontal_shift in search_space['horizontal']:
            for scaling_factor in search_space['scaling_factor']:
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'vertical': vertical_shift,
                    'horizontal': horizontal_shift,
                    'scaling': scaling_factor,
                    'best_sim': f"{best_similarity:.4f}",
                    'metric': f"{metric}{'_bin' if binarize else ''}",
                    'trim': trim_threshold
                })
                
                # Apply transformations
                transformed_intervals = transform_note_intervals(
                    note_intervals,
                    vertical_shift,
                    horizontal_shift,
                    scaling_factor
                )
                
                if has_negative_times(transformed_intervals):
                    continue
                
                # Convert to pitch contour
                midi_function = create_midi_function(transformed_intervals)
                midi_contour = create_midi_contour(midi_function, time_stamps)
                pitch_contour = convert_midi_to_pitch(midi_contour)
                
                # Calculate similarity metrics
                metrics = {
                    'cosine': {
                        'normal': compute_similarity_metric(
                            reference_contour, pitch_contour, 
                            metric='cosine', binarize=False
                        ),
                        'binary': compute_similarity_metric(
                            reference_contour, pitch_contour, 
                            metric='cosine', binarize=True
                        )
                    },
                    'mse': {
                        'normal': compute_similarity_metric(
                            reference_contour, pitch_contour, 
                            metric='mse', binarize=False
                        ),
                        'binary': compute_similarity_metric(
                            reference_contour, pitch_contour, 
                            metric='mse', binarize=True
                        )
                    },
                    'mae': {
                        'normal': compute_similarity_metric(
                            reference_contour, pitch_contour, 
                            metric='mae', binarize=False
                        ),
                        'binary': compute_similarity_metric(
                            reference_contour, pitch_contour, 
                            metric='mae', binarize=True
                        )
                    },
                    'mean_difference': compute_similarity_metric(
                        reference_contour, pitch_contour, 
                        metric='mean_difference', binarize=False
                    )
                }
                
                # Select appropriate similarity based on parameters
                if metric == 'cosine':
                    similarity = metrics['cosine']['binary'] if binarize else metrics['cosine']['normal']
                elif metric == 'mse':
                    similarity = metrics['mse']['binary'] if binarize else metrics['mse']['normal']
                elif metric == 'mae':
                    similarity = metrics['mae']['binary'] if binarize else metrics['mae']['normal']
                else:
                    similarity = metrics['mean_difference']
                
                # Record search results
                search_record.append((
                    vertical_shift, 
                    horizontal_shift, 
                    scaling_factor,
                    metrics['cosine']['normal'], 
                    metrics['cosine']['binary'],
                    metrics['mse']['normal'], 
                    metrics['mse']['binary'],
                    metrics['mae']['normal'], 
                    metrics['mae']['binary'],
                    metrics['mean_difference'],
                    trim_threshold  
                ))
                
                # Update best parameters if new best found
                if (maximize and similarity > best_similarity) or (not maximize and similarity < best_similarity):
                    best_similarity = similarity
                    best_parameters = (vertical_shift, horizontal_shift, scaling_factor)
                    
    progress_bar.close()
    return best_parameters, best_similarity
