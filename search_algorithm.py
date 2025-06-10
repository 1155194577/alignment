import numpy as np 
import pandas as pd
import mido
from tqdm import tqdm
from midi_processing import transform_note_event_interval,convert_note_event_intervals_to_midi_function,convert_midi_function_to_midi_contour,convert_midi_contour_to_pitch_contour
from util import compute_similarity_metric
    
# Helper function to validate events
def has_negative_times(events):
    """Check if any event has negative time values"""
    return any(event['start_time'] < 0 or event['end_time'] < 0 for event in events)

def brute_force_search(reference_contour, time, midi_events, search_space, search_record, 
                      metric='cosine', binarise=False, maximize=True):
    best_similarity = float('-inf') if maximize else float('inf')
    best_parameters = None
    total = len(search_space['vertical']) * len(search_space['horizontal']) * len(search_space['scaling_factor'])
    progress_bar = tqdm(total=total, desc="Brute Force Search")
    
    for vertical_displacement in search_space['vertical']:
        for horizontal_displacement in search_space['horizontal']:
            for scaling_factor in search_space['scaling_factor']: 
                progress_bar.update(1)
                progress_bar.set_postfix({ 
                    'vertical': vertical_displacement,
                    'horizontal': horizontal_displacement,
                    'scaling': scaling_factor,
                    'best_sim': f"{best_similarity:.4f}",
                    'metric': f"{metric}{'_bin' if binarise else ''}"
                })
                
                transformed_note_event_interval = transform_note_event_interval(midi_events, vertical_displacement, horizontal_displacement, scaling_factor)
                if has_negative_times(transformed_note_event_interval):
                    continue
                transformed_midi_function = convert_note_event_intervals_to_midi_function(transformed_note_event_interval)
                transformed_midi_contour = convert_midi_function_to_midi_contour(transformed_midi_function, time)
                transformed_pitch_contour = convert_midi_contour_to_pitch_contour(transformed_midi_contour)
                cosine_similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=False)
                binary_cosine_similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=True)
                mse = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='mse', binarise=False)
                binary_mse = compute_similarity_metric(reference_contour,transformed_pitch_contour,metric='mse',binarise=True)
                mae = compute_similarity_metric(reference_contour,transformed_pitch_contour,metric="mae",binarise=False)
                binary_mae = compute_similarity_metric(reference_contour,transformed_pitch_contour,metric="mae",binarise=True)
                mean_difference = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='mean_difference', binarise=False)
                # Select similarity based on metric and binarise parameters
                if metric == 'cosine':
                    similarity = binary_cosine_similarity if binarise else cosine_similarity
                elif metric == 'mse':
                    similarity = binary_mse if binarise else mse
                elif metric == 'mae':
                    similarity = binary_mae if binarise else mae
                elif metric == 'mean_difference':
                    similarity = mean_difference
                else:
                    similarity = cosine_similarity  # Default to cosine
                # Record the search parameters and similarity
                search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, cosine_similarity,binary_cosine_similarity,mse, binary_mse,mae,binary_mae,mean_difference))
                if (maximize and similarity > best_similarity) or (not maximize and similarity < best_similarity):
                    improvement = similarity - best_similarity if maximize else best_similarity - similarity
                    best_similarity = similarity
                    # progress_bar.write(f"Best similarity updated! {best_similarity:.4f} improvement: {improvement:.4f} parameter ({vertical_displacement}, {horizontal_displacement}, {scaling_factor})")
                    best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
                    
    progress_bar.close()
    return best_parameters, best_similarity

def random_search(reference_contour, time, midi_events, search_space, search_record, n_iter=100,
                 metric='cosine', binarise=False):
    best_similarity = float('-inf')
    best_parameters = None
    
    progress_bar = tqdm(total=n_iter, desc="Random Search")
    for i in range(n_iter):
        # Randomly select parameters from search space
        vertical_displacement = np.random.choice(search_space['vertical'])
        horizontal_displacement = np.random.choice(search_space['horizontal'])
        scaling_factor = np.random.choice(search_space['scaling_factor'])
        
        progress_bar.update(1)
        progress_bar.set_postfix({
            'vertical': vertical_displacement,
            'horizontal': horizontal_displacement,
            'scaling': scaling_factor,
            'best_sim': f"{best_similarity:.4f}",
            'metric': f"{metric}{'_bin' if binarise else ''}"
        })
        
        transformed_note_event_interval = transform_note_event_interval(midi_events, vertical_displacement, horizontal_displacement, scaling_factor)
        if has_negative_times(transformed_note_event_interval):
            continue
        transformed_midi_function = convert_note_event_intervals_to_midi_function(transformed_note_event_interval)
        transformed_midi_contour = convert_midi_function_to_midi_contour(transformed_midi_function, time)
        transformed_pitch_contour = convert_midi_contour_to_pitch_contour(transformed_midi_contour)
        similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, 
                                             metric=metric, binarise=binarise)
        
        search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, similarity))
        if similarity > best_similarity:
            improvement = similarity - best_similarity
            best_similarity = similarity
            progress_bar.write(f"Best similarity updated! {best_similarity:.4f} improvement: {improvement:.4f} parameter ({vertical_displacement}, {horizontal_displacement}, {scaling_factor})")
            best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
            
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
                             dense_space, search_record,
                             metric=metric, binarise=binarise)
