import numpy as np 
import pandas as pd
import mido
from midi_processing import transform_note_event_interval,convert_note_event_intervals_to_midi_function,convert_midi_function_to_midi_contour,convert_midi_contour_to_pitch_contour
from util import compute_similarity_metric
    
# Helper function to validate events
def has_negative_times(events):
    """Check if any event has negative time values"""
    return any(event['start_time'] < 0 or event['end_time'] < 0 for event in events)

def brute_force_search(reference_contour, time, midi_events, search_space, search_record):
    best_similarity = float('-inf')
    best_parameters = None
    total = len(search_space['vertical']) * len(search_space['horizontal']) * len(search_space['scaling_factor'])
    count = 0
    for vertical_displacement in search_space['vertical']:
        for horizontal_displacement in search_space['horizontal']:
          for scaling_factor in search_space['scaling_factor']: 
                count += 1
                print(f"Processing {count}/{total} (Vertical: {vertical_displacement}, Horizontal: {horizontal_displacement}, Scaling factor: {scaling_factor})")
                
                transformed_note_event_interval = transform_note_event_interval(midi_events, vertical_displacement, horizontal_displacement, scaling_factor)
                if has_negative_times(transformed_note_event_interval):
                    print("Skipping due to negative times in transformed intervals")
                    continue
                transformed_midi_function = convert_note_event_intervals_to_midi_function(transformed_note_event_interval)
                transformed_midi_contour = convert_midi_function_to_midi_contour(transformed_midi_function, time)
                transformed_pitch_contour = convert_midi_contour_to_pitch_contour(transformed_midi_contour)
                cosine_similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=False)
                binary_cosine_similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=True)
                negative_mse = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='negative-mse', binarise=False)
                
                similarity = cosine_similarity
                # Record the search parameters and similarity
                search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, cosine_similarity,binary_cosine_similarity, negative_mse))
                if similarity > best_similarity:
                    improvement = similarity - best_similarity
                    best_similarity = similarity
                    print("Best similarity updated!", best_similarity, "improvement:", improvement, "parameter", (vertical_displacement, horizontal_displacement, scaling_factor))
                    best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
    return best_parameters, best_similarity

def random_search(reference_contour, time, midi_events, search_space, search_record, n_iter=100):
    best_similarity = float('-inf')
    best_parameters = None
    
    for i in range(n_iter):
        # Randomly select parameters from search space
        vertical_displacement = np.random.choice(search_space['vertical'])
        horizontal_displacement = np.random.choice(search_space['horizontal'])
        scaling_factor = np.random.choice(search_space['scaling_factor'])
        
        print(f"Processing iteration {i+1}/{n_iter} (Vertical: {vertical_displacement}, Horizontal: {horizontal_displacement}, Scaling factor: {scaling_factor})")
        
        transformed_note_event_interval = transform_note_event_interval(midi_events, vertical_displacement, horizontal_displacement, scaling_factor)
        if has_negative_times(transformed_note_event_interval):
                    print("Skipping due to negative times in transformed intervals")
                    continue
        transformed_midi_function = convert_note_event_intervals_to_midi_function(transformed_note_event_interval)
        transformed_midi_contour = convert_midi_function_to_midi_contour(transformed_midi_function, time)
        transformed_pitch_contour = convert_midi_contour_to_pitch_contour(transformed_midi_contour)
        similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=False)
        
        search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, similarity))
        if similarity > best_similarity:
            improvement = similarity - best_similarity
            best_similarity = similarity
            print("Best similarity updated!", best_similarity, "improvement:", improvement, "parameter", (vertical_displacement, horizontal_displacement, scaling_factor))
            best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
    
    return best_parameters, best_similarity

def hybrid_search(reference_contour, time, midi_events, search_space, search_record, n_random=100, region_fraction=0.2):
    """
    Hybrid search combining random exploration and local brute-force optimization
    
    Parameters:
    n_random: Number of random samples in phase 1
    region_fraction: Fraction of parameter range to search around best point (±value)
    """
    # Phase 1: Random search to find promising region
    best_params, best_sim = random_search(reference_contour, time, midi_events, 
                                         search_space, search_record, n_random)
    
    if best_params is None:
        return None, best_sim
    
    # Unpack best parameters
    v_best, h_best, s_best = best_params
    
    # Define promising region boundaries (clamped to original search space)
    v_min = max(min(search_space['vertical']), 
               v_best - region_fraction * (max(search_space['vertical']) - min(search_space['vertical'])))
    v_max = min(max(search_space['vertical']), 
               v_best + region_fraction * (max(search_space['vertical']) - min(search_space['vertical'])))
    
    h_min = max(min(search_space['horizontal']), 
               h_best - region_fraction * (max(search_space['horizontal']) - min(search_space['horizontal'])))
    h_max = min(max(search_space['horizontal']), 
               h_best + region_fraction * (max(search_space['horizontal']) - min(search_space['horizontal'])))
    
    s_min = max(min(search_space['scaling_factor']), 
               s_best - region_fraction * (max(search_space['scaling_factor']) - min(search_space['scaling_factor'])))
    s_max = min(max(search_space['scaling_factor']), 
               s_best + region_fraction * (max(search_space['scaling_factor']) - min(search_space['scaling_factor'])))
    
    # Create dense search space for phase 2
    dense_space = {
        'vertical': np.linspace(v_min, v_max, num=50),
        'horizontal': np.linspace(h_min, h_max, num=50),
        'scaling_factor': np.linspace(s_min, s_max, num=50)
    }
    
    print(f"\nStarting phase 2: Brute-force search in promising region")
    print(f"Vertical: {v_min:.2f} to {v_max:.2f}")
    print(f"Horizontal: {h_min:.2f} to {h_max:.2f}")
    print(f"Scaling: {s_min:.2f} to {s_max:.2f}")
    
    # Phase 2: Brute-force search in promising region
    return brute_force_search(reference_contour, time, midi_events, 
                             dense_space, search_record)
