import numpy as np 
import pandas as pd
from midi_util import convert_midi_to_events,get_midi_bpm,normalize_midi_event,convert_events_to_midi_function,convert_event_to_midi
import matplotlib.pyplot as plt
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

def transform_midi_events(midi_events, vertical_displacement, horizontal_displacement, beat_per_minutes):
        """
        Apply vertical (pitch), horizontal (timing), and tempo (BPM) transformations to MIDI events.
        Args:
            midi_events (list of dict): Each dict has 'start_time', 'end_time', 'note', 'velocity'. Assumed to be normalized.
            vertical_displacement (float): Semitone shift for pitch.
            horizontal_displacement (float): Time shift in milliseconds.
            beat_per_minutes (float): Target BPM for tempo scaling.
        Returns:
            list of dict: Transformed MIDI events.
        """
        # Assume original BPM is stored in each event or passed separately; here we use 120 as default
        tempo_scale = beat_per_minutes

        transformed_events = []
        for event in midi_events:
            new_event = event.copy()
            # Pitch shift
            new_event['note'] = event['note'] + vertical_displacement
            # Time shift and tempo scaling
            new_event['start_time'] = (event['start_time']*tempo_scale) + horizontal_displacement
            new_event['end_time'] = (event['end_time'] *tempo_scale) + horizontal_displacement
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

def compute_normalized_cross_correlation(contour_a, contour_b):
    """
    Compute the normalized cross-correlation between two contours. 
    Parameters: 
    contour_a (np array): The first contour. 
    contour_b (np array): The second contour. 
    Returns:
    float: The normalized cross-correlation value.  between -1 and 1
    """
    dot_product = np.dot(contour_a, contour_b) 
    norm_a = np.linalg.norm(contour_a) 
    norm_b = np.linalg.norm(contour_b) 
    if norm_a == 0 or norm_b == 0: 
        return 0.0 
    else: 
        return dot_product / (norm_a * norm_b) 
    
def compute_similarity_metric(reference_contour, candidate_midi_contour): 
    ncc = compute_normalized_cross_correlation(reference_contour, candidate_midi_contour) 
    return ncc

def convert_search_record_to_dataframe(search_record): 
    """
    Convert the search record to a pandas DataFrame for easier analysis and visualization.
    Parameters:
    search_record (list of tuples): Each tuple contains (vertical_displacement, horizontal_displacement, beat_per_minutes, similarity).
    Returns:
    pd.DataFrame: A DataFrame with columns for vertical displacement, horizontal displacement, BPM, and similarity.
    """
    df = pd.DataFrame(search_record, columns=['vertical_displacement', 'horizontal_displacement', 'beat_per_minutes', 'similarity'])
    return df

def brute_force_search(reference_contour, time, midi_events, search_space, search_record): 
    best_similarity = float('-inf')
    best_parameters = None
    total = len(search_space['vertical']) * len(search_space['horizontal']) * len(search_space['beat_per_minutes'])
    count = 0
    for vertical_displacement in search_space['vertical']:
        for horizontal_displacement in search_space['horizontal']:
          for beat_per_minutes in search_space['beat_per_minutes']:
                count += 1
                print(f"Processing {count}/{total} (Vertical: {vertical_displacement}, Horizontal: {horizontal_displacement}, BPM: {beat_per_minutes})")
                transformed_midi_events = transform_midi_events(midi_events, vertical_displacement, horizontal_displacement, beat_per_minutes)
                transformed_midi_contour = convert_events_to_midi_contour(transformed_midi_events, time)
                similarity = compute_similarity_metric(reference_contour, transformed_midi_contour)
                print(f"Similarity: {similarity}")
                # Record the search parameters and similarity
                search_record.append((vertical_displacement, horizontal_displacement, beat_per_minutes, similarity))
                if similarity > best_similarity:
                    improvement = similarity - best_similarity
                    best_similarity = similarity
                    print("Best similarity updated!", best_similarity, "improvement:", improvement)
                    best_parameters = (vertical_displacement, horizontal_displacement, beat_per_minutes)
    return best_parameters, best_similarity


if __name__ == "__main__": 
    reference_contour_path = "signal_a_contour_tony.csv" 
    reference_contour, time = load_reference_contour(reference_contour_path) 
    print("Loaded reference contour:", reference_contour, reference_contour.shape)
    print("Loaded time values:", time, time.shape)
    
    # # plot_pitch_contour(time, reference_contour, start=0, end=10000) 

    midi_file_path = "signal_b.mid"
    midi_events = convert_midi_to_events(midi_file_path) 
    print(midi_events)
    midi_bpm = get_midi_bpm(midi_file_path)
    normalized_midi_events = normalize_midi_event(midi_events,midi_bpm)
    # midi_key_signature = get_midi_key_signature(midi_file_path) 
    search_space = {
        'vertical': np.array([0]),
        'horizontal': np.arange(-5, 5, 1),  
        'beat_per_minutes':[midi_bpm + offset for offset in np.arange(-20, 20.5, 0.5)]
    }
    search_record = []
    best_parameters, best_similarity = brute_force_search(reference_contour, time, normalized_midi_events, search_space, search_record)
    print("Best parameters:", best_parameters)
    print("Best cost:", best_similarity)
    
    df = convert_search_record_to_dataframe(search_record)  
    df.to_csv("search_results.csv", index=False)
    print("apply best parameters to midi and save as midi file")
    vertical_displacement, horizontal_displacement, beat_per_minutes = best_parameters
    transformed_midi_events = transform_midi_events(normalized_midi_events, vertical_displacement, horizontal_displacement, beat_per_minutes) 
    saved_midi_file_path = "optimized_signal_b.mid" 
    convert_event_to_midi(transformed_midi_events, saved_midi_file_path)
    print(f"Transformed MIDI events saved to {saved_midi_file_path}")