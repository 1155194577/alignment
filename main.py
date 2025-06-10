from search_algorithm import brute_force_search, random_search
from midi_processing import convert_midi_to_note_event_intervals, convert_tick_to_seconds, get_midi_bpm, get_midi_tick_per_second, convert_note_event_intervals_to_mid, transform_note_event_interval, convert_seconds_to_tick
from audio_processing import load_reference_contour, trim_times_below_threshold
import mido
from util import convert_search_record_to_dataframe
import numpy as np

def load_reference_data(path: str, trim_factor: float = 1.0) -> tuple:
    """Load and trim reference contour data from CSV file."""
    contour, times = load_reference_contour(path)
    return trim_times_below_threshold(times, contour, times[-1] * trim_factor)

def load_and_process_midi(path: str) -> tuple:
    """Load MIDI file and extract note events in seconds."""
    mid = mido.MidiFile(path)
    bpm = get_midi_bpm(mid)
    tick_rate = get_midi_tick_per_second(mid)
    events_ticks = convert_midi_to_note_event_intervals(mid)
    events_sec = convert_tick_to_seconds(events_ticks, tick_rate, bpm)
    return bpm, tick_rate, events_sec

def define_search_space(
    base_bpm: float,
    vertical_min: float = -12.0,
    vertical_max: float = 12.0,
    vertical_step: float = 1.0,
    horizontal_min: float = 0.0,
    horizontal_max: float = 2.0,
    horizontal_step: float = 0.1,
    bpm_offset_min: float = -10.0,
    bpm_offset_max: float = 10.0,
    bpm_offset_step: float = 1.0
) -> dict:
    """
    Generate configurable parameter ranges for search algorithm.
    
    Args:
        base_bpm: Reference BPM for scaling factor calculation
        vertical_min: Minimum pitch shift in semitones
        vertical_max: Maximum pitch shift in semitones
        vertical_step: Pitch shift step size
        horizontal_min: Minimum time shift in seconds
        horizontal_max: Maximum time shift in seconds
        horizontal_step: Time shift step size
        bpm_offset_min: Minimum BPM offset from base
        bpm_offset_max: Maximum BPM offset from base
        bpm_offset_step: BPM offset step size
        
    Returns:
        Dictionary containing parameter search space
    """
    # Generate vertical displacement range
    vertical_range = np.arange(vertical_min, vertical_max + vertical_step, vertical_step)
    
    # Generate horizontal displacement range
    horizontal_range = np.arange(horizontal_min, horizontal_max + horizontal_step, horizontal_step)
    
    # Generate BPM offsets and filter valid values
    bpm_offsets = np.arange(bpm_offset_min, bpm_offset_max + bpm_offset_step, bpm_offset_step)
    valid_bpms = [base_bpm + offset for offset in bpm_offsets if (base_bpm + offset) > 0]
    
    # Calculate scaling factors from valid BPMs
    scaling_factors = np.unique(
        np.concatenate([
            np.array([base_bpm / bpm for bpm in valid_bpms]),
            np.array([bpm / base_bpm for bpm in valid_bpms])
        ])
    )
    
    return {
        'vertical': vertical_range,
        'horizontal': horizontal_range,
        'scaling_factor': scaling_factors
    }

def run_search(strategy: str, ref_contour, times, note_events, search_space, similarity_metric="cosine", binarise=False, maximize=True) -> tuple:
    """Execute search algorithm and return best parameters."""
    record = []
    if strategy == 'brute_force':
        best_params, similarity = brute_force_search(
            ref_contour, times, note_events, search_space, record, metric=similarity_metric, binarise=binarise, maximize=maximize
        )    
    else:  # Default to random search
        best_params, similarity = random_search(
            ref_contour, times, note_events, search_space, record, n_iter=10, metric=similarity_metric, binarise=binarise
        )
    return best_params, similarity, record

def save_optimized_midi(params, note_events, tick_rate, bpm, output_path: str):
    """Apply transformations and save optimized MIDI file."""
    v_disp, h_disp, scaling = params
    optimized = transform_note_event_interval(note_events, v_disp, h_disp, scaling)
    optimized_ticks = convert_seconds_to_tick(optimized, tick_rate, bpm)
    mid = convert_note_event_intervals_to_mid(optimized_ticks, bpm, tick_rate)
    mid.save(output_path)

def find_best_params(search_space,ref_contour, note_events, times, similarity_metric="cosine", binarise=False):
    
    is_maximize = True if similarity_metric in ["cosine"] else False

    best_params, similarity, record = run_search(
        'brute_force', ref_contour, times, note_events, search_space, similarity_metric, binarise, maximize=is_maximize
    )
    print(f"Best parameters: {best_params}")
    print(f"Best similarity: {similarity}")
    return best_params,record
def main():
    trim_factor = 0.1
    ref_contour, times = load_reference_data("./input/ep_030_pitch.csv", trim_factor)
    midi_bpm, tick_rate, note_events = load_and_process_midi("./input/ep_030.mid")
    vertical_search_space = define_search_space(
        midi_bpm,
        vertical_min=-20.0,    # e.g., limit pitch shift to -6 semitones
        vertical_max=0,     # and +6 semitones
        vertical_step=1,
        horizontal_min=0,   # time shift from 0
        horizontal_max=0,   # to 1 second
        horizontal_step=1, # finer time shift steps
        bpm_offset_min=0,    
        bpm_offset_max=0,  
        bpm_offset_step=1   # finer BPM steps
    )
    horizontal_bpm_search_space = define_search_space(
        midi_bpm,
        vertical_min=0,    # e.g., limit pitch shift to -6 semitones
        vertical_max=0,     # and +6 semitones
        vertical_step=1,
        horizontal_min=0,   # time shift from 0
        horizontal_max=3,   # to 1 second
        horizontal_step=0.1, # finer time shift steps
        bpm_offset_min=-10,    
        bpm_offset_max=10,  
        bpm_offset_step=1   
    )
    intermediate_params, record1  = find_best_params(horizontal_bpm_search_space, ref_contour, note_events, times, similarity_metric="cosine", binarise=False)
    save_optimized_midi(
        intermediate_params, note_events, tick_rate, midi_bpm/intermediate_params[2],
        "./output/intermediate_params.mid"
    )
    convert_search_record_to_dataframe(record1).to_csv("./output/first_search.csv", index=False)
    transformed_note_events = transform_note_event_interval(note_events, *intermediate_params)

    best_params, record2 = find_best_params(vertical_search_space, ref_contour, transformed_note_events, times, similarity_metric="mae", binarise=False)    
    save_optimized_midi(
        best_params, transformed_note_events, tick_rate, midi_bpm/intermediate_params[2],
        "./output/best_params.mid"
    )
    convert_search_record_to_dataframe(record2).to_csv("./output/second_search.csv", index=False)
    final_note_events = transform_note_event_interval(note_events, *best_params) 

    print(f"Final optimized parameters: {best_params}")
    
if __name__ == "__main__":
    main()
