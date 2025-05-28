import numpy as np 
import pandas as pd
from midi_util import convert_midi_to_events,get_midi_bpm,normalize_midi_event,convert_events_to_midi_function,convert_event_to_midi
import matplotlib.pyplot as plt
import mido
def load_reference_contour(file_path):
    """
    Load the reference contour from a CSV file.
    Parameters:
    file_path (str): The path to the CSV file.
    Returns:
    a tuple: A tuple containing the reference contour (np.array) and the corresponding time values (np.array).
    """
    df = pd.read_csv(file_path)
    time = df['time'].to_numpy()
    time = time * 1000  # Convert to milliseconds
    pitches = df['value'].to_numpy()
    return pitches, time

def transform_midi_events(midi_events, vertical_displacement, horizontal_displacement, scaling_factor):
    """
    Apply pitch shift, timing shift, and tempo scaling to MIDI events.
    Args:
        midi_events (list of dict): Each dict has 'start_time', 'end_time', 'note', 'velocity'.
        vertical_displacement (float): Semitone shift for pitch.
        horizontal_displacement (float): Time shift in milliseconds.
        scaling_factor (float): Factor to scale timing (tempo adjustment).
    Returns:
        list of dict: Transformed MIDI events.
    """
    transformed_events = []
    for event in midi_events:
        new_event = event.copy()
        # Apply pitch shift
        new_event['note'] = event['note'] + vertical_displacement
        # Apply time shift and tempo scaling
        new_event['start_time'] = (event['start_time'] * scaling_factor) + horizontal_displacement
        new_event['end_time'] = (event['end_time'] * scaling_factor) + horizontal_displacement
        transformed_events.append(new_event)
    return transformed_events

def convert_events_to_midi_contour(events,time):  
    midi_function = convert_events_to_midi_function(events)
    midi_contour = []
    for time_value in time:
        note_number = midi_function(time_value) 
        if len(note_number) > 0:
            note_number = note_number[0]
            midi_contour.append(note_number)
        else: 
            note_number = 0 
            midi_contour.append(note_number)
    return np.array(midi_contour)

def compute_normalized_cross_correlation(contour_a, contour_b, binarize=True):
    """
    Compute the normalized cross-correlation between two contours. 
    Parameters: 
    contour_a (np array): The first contour. 
    contour_b (np array): The second contour. 
    binarize (bool): Whether to binarize contours before comparison (default: True)
    Returns:
    float: The normalized cross-correlation value.  between -1 and 1
    """
    if binarize:
        binarize_fn = lambda contour: np.where(contour > 0, 1, 0)
        contour_a = binarize_fn(contour_a)
        contour_b = binarize_fn(contour_b)
    # print("shape,", contour_a.shape, contour_b.shape)
    dot_product = np.dot(contour_a, contour_b) 
    norm_a = np.linalg.norm(contour_a) 
    norm_b = np.linalg.norm(contour_b) 
    if norm_a == 0 or norm_b == 0: 
        return 0.0 
    else: 
        return dot_product / (norm_a * norm_b) 
    
def compute_similarity_metric(reference_contour, candidate_midi_contour, binarize=True): 
    ncc = compute_normalized_cross_correlation(reference_contour, candidate_midi_contour, binarize) 
    return ncc

def convert_search_record_to_dataframe(search_record): 
    """
    Convert the search record to a pandas DataFrame for easier analysis and visualization.
    Parameters:
    search_record (list of tuples): Each tuple contains (vertical_displacement, horizontal_displacement, scaling_factor, similarity).
    Returns:
    pd.DataFrame: A DataFrame with columns for vertical displacement, horizontal displacement, scaling factor, and similarity.
    """
    df = pd.DataFrame(search_record, columns=['vertical_displacement', 'horizontal_displacement', 'scaling_factor', 'similarity'])
    return df

def brute_force_search(reference_contour, time, midi_events, search_space, search_record, binarize=True): 
    best_similarity = float('-inf')
    best_parameters = None
    total = len(search_space['vertical']) * len(search_space['horizontal']) * len(search_space['scaling_factor'])
    count = 0
    for vertical_displacement in search_space['vertical']:
        for horizontal_displacement in search_space['horizontal']:
          for scaling_factor in search_space['scaling_factor']:
                count += 1
                # print(f"Processing {count}/{total} (Vertical: {vertical_displacement}, Horizontal: {horizontal_displacement}, Scaling factor: {scaling_factor})")
                transformed_midi_events = transform_midi_events(midi_events, vertical_displacement, horizontal_displacement, scaling_factor)
                transformed_midi_contour = convert_events_to_midi_contour(transformed_midi_events, time)
                similarity = compute_similarity_metric(reference_contour, transformed_midi_contour, binarize=binarize)
                # print(f"Similarity: {similarity}")
                # Record the search parameters and similarity
                search_record.append((vertical_displacement, horizontal_displacement, scaling_factor, similarity))
                if similarity > best_similarity:
                    improvement = similarity - best_similarity
                    best_similarity = similarity
                    print("Best similarity updated!", best_similarity, "improvement:", improvement, "parameter", (vertical_displacement, horizontal_displacement, scaling_factor))
                    best_parameters = (vertical_displacement, horizontal_displacement, scaling_factor)
    return best_parameters, best_similarity


if __name__ == "__main__":
    # Configuration flag for binarization
    BINARIZE = False  # Set to False to disable contour binarization
    source_folder = "./trimmed_24" 
    reference_contour_name = "vocals_tony.csv" 
    reference_contour_path = f"{source_folder}/{reference_contour_name}"
    reference_contour, time = load_reference_contour(reference_contour_path) 
    print("Loaded reference contour:", reference_contour, reference_contour.shape)
    print("Loaded time values:", time, time.shape)
    midi_file_name = "annotation.mid"
    midi_file_path = f"{source_folder}/{midi_file_name}"
    midi_events = convert_midi_to_events(midi_file_path) 
    print(midi_events)
    midi_bpm = get_midi_bpm(midi_file_path)
    # normalized_midi_events = normalize_midi_event(midi_events,midi_bpm)
    # midi_key_signature = get_midi_key_signature(midi_file_path) 

    midi_bpm_range = [midi_bpm + offset for offset in np.arange(-50, 51, 1)]
    scaling_factor_range1 = np.array([midi_bpm/x for x in midi_bpm_range])
    scaling_factor_range2 = np.array([x/midi_bpm for x in midi_bpm_range])
    # Choose which scaling factor range to use (union of both)
    scaling_factor_range = np.unique(np.concatenate([scaling_factor_range1, scaling_factor_range2]))
    print(scaling_factor_range)
    search_space = {
        'vertical': np.array([0]),
        'horizontal': np.arange(-0*1000, 5*1000, 100),  
        'scaling_factor': scaling_factor_range, 
    }
    search_record = []
    best_parameters, best_similarity = brute_force_search(reference_contour, time, midi_events, search_space, search_record, binarize=BINARIZE)
    print("Best parameters:", best_parameters)
    print("Best cost:", best_similarity)
    df = convert_search_record_to_dataframe(search_record)  
    df.to_csv("search_results.csv", index=False)
    print("apply best parameters to midi and save as midi file")
    vertical_displacement, horizontal_displacement, scaling_factor = best_parameters
    transformed_midi_events = transform_midi_events(midi_events, vertical_displacement, horizontal_displacement, scaling_factor) 
    saved_midi_file_name = "optimized_signal_b.mid" 
    saved_midi_file_path = f"{source_folder}/{saved_midi_file_name}" 
     
    ## convert scaling_factor to bpm 
    beat_per_minutes = midi_bpm / scaling_factor
    print(f"Best BPM: {beat_per_minutes}, scaling factor: {scaling_factor}")
    ## convert bpm to tempo 
    best_tempo =  mido.bpm2tempo(beat_per_minutes)
    convert_event_to_midi(transformed_midi_events, saved_midi_file_path, best_tempo) 
    print(f"Transformed MIDI events saved to {saved_midi_file_path}")
    # Convert BPM to MIDI tempo format 
# convert_event_to_midi(transformed_midi_events, "transformed_signal_b.mid",tempo)  # Save the transformed MIDI events to a file
