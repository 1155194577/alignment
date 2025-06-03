import numpy as np 
import pandas as pd
import mido
from midi_processing import transform_note_event_interval,convert_note_event_intervals_to_midi_function,convert_midi_function_to_midi_contour,convert_midi_contour_to_pitch_contour
from util import compute_similarity_metric
    

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
                transformed_midi_function = convert_note_event_intervals_to_midi_function(transformed_note_event_interval)
                transformed_midi_contour = convert_midi_function_to_midi_contour(transformed_midi_function, time)
                transformed_pitch_contour = convert_midi_contour_to_pitch_contour(transformed_midi_contour)
                similarity = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='cosine', binarise=False)
                # negative_mse = compute_similarity_metric(reference_contour, transformed_pitch_contour, metric='negative-mse', binarise=False)
                # Record the search parameters and similarity
                search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, similarity))
                if similarity > best_similarity:
                    improvement = similarity - best_similarity
                    best_similarity = similarity
                    print("Best similarity updated!", best_similarity, "improvement:", improvement, "parameter", (vertical_displacement, horizontal_displacement, scaling_factor))
                    best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
    return best_parameters, best_similarity
