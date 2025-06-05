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

def define_search_space(base_bpm: float) -> dict:
    """Generate parameter ranges for search algorithm."""
    horizontal_range = np.arange(0, 2, 0.1)
    bpm_range = [base_bpm + offset for offset in np.arange(-10, 10, 1)]
    scaling_factors = np.unique(np.concatenate([
        [base_bpm/x for x in bpm_range],
        [x/base_bpm for x in bpm_range]
    ]))
    return {
        'vertical': [0],
        'horizontal': horizontal_range,
        'scaling_factor': scaling_factors
    }

def run_search(strategy: str, ref_contour, times, note_events, search_space) -> tuple:
    """Execute search algorithm and return best parameters."""
    record = []
    if strategy == 'brute_force':
        best_params, similarity = brute_force_search(
            ref_contour, times, note_events, search_space, record
        )
    else:  # Default to random search
        best_params, similarity = random_search(
            ref_contour, times, note_events, search_space, record, n_iter=10
        )
    return best_params, similarity, record

def save_optimized_midi(params, note_events, tick_rate, bpm, output_path: str):
    """Apply transformations and save optimized MIDI file."""
    v_disp, h_disp, scaling = params
    optimized = transform_note_event_interval(note_events, v_disp, h_disp, scaling)
    optimized_ticks = convert_seconds_to_tick(optimized, tick_rate, bpm)
    mid = convert_note_event_intervals_to_mid(optimized_ticks, bpm, tick_rate)
    mid.save(output_path)

def main(trim_factor: float = 0.7):
    # 1. Load reference contour data
    ref_contour, times = load_reference_data("./john/ep_030_pitch.csv", trim_factor)
    
    # 2. Process MIDI file
    midi_bpm, tick_rate, note_events = load_and_process_midi("./john/ep_030.mid")
    
    # 3. Define search space
    search_space = define_search_space(midi_bpm)
    
    # 4. Run search (default to brute-force)
    best_params, similarity, record = run_search(
        'brute_force', ref_contour, times, note_events, search_space
    )
    
    # Output results
    print(f"Best parameters: {best_params}")
    print(f"Best similarity: {similarity}")
    print(f"Original BPM: {midi_bpm}, New BPM: {midi_bpm / best_params[2]}")
    
    # Save search results
    convert_search_record_to_dataframe(record).to_csv("search_results.csv", index=False)
    
    # 5. Save optimized MIDI
    save_optimized_midi(
        best_params, note_events, tick_rate, midi_bpm,
        "./johnny_simple_midi/Johnny_simpleMIDI_optimized.mid"
    )
    return trim_factor, best_params , similarity
if __name__ == "__main__":
    results = []
    for trim_factor in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        result = main(trim_factor)
        results.append(result)
    print("All results:")
    for trim_factor, best_params, similarity in results:
        print(f"Trim factor: {trim_factor}, Best params: {best_params}, Similarity: {similarity:.4f}")
