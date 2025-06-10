from typing import Tuple, Dict, List
from search_algorithm import brute_force_search
from midi_processing import (
    convert_midi_to_note_intervals,
    convert_ticks_to_seconds,
    get_midi_tempo,
    get_midi_ticks_per_second,
    create_midi_from_note_intervals,
    transform_note_intervals,
    convert_seconds_to_ticks
)
from audio_processing import load_reference_contour, trim_contour_data
import mido
from util import convert_search_record_to_dataframe
import numpy as np

# Configuration constants
TRIM_THRESHOLD = 0.1

# Directory paths
INPUT_DIRECTORY = "./input"
OUTPUT_DIRECTORY = "./output"

# File names
REFERENCE_PITCH_FILE = "ep_030_pitch.csv"
MIDI_INPUT_FILE = "ep_030.mid"

# First stage search parameters (vertical alignment)
VERTICAL_SEARCH_CONFIG = {
    "vertical_min": -20.0,    # Minimum pitch shift in semitones
    "vertical_max": 0,        # Maximum pitch shift in semitones
    "vertical_step": 1,       # Pitch shift step size
    "horizontal_min": 0,      # Horizontal shift (fixed at 0 for vertical search)
    "horizontal_max": 0,
    "horizontal_step": 1,
    "bpm_offset_min": 0,      # BPM offset (fixed at 0 for vertical search)
    "bpm_offset_max": 0,
    "bpm_offset_step": 1
}

# Second stage search parameters (horizontal alignment and BPM)
HORIZONTAL_BPM_SEARCH_CONFIG = {
    "vertical_min": 0,        # Vertical shift (fixed at 0 for horizontal search)
    "vertical_max": 0,
    "vertical_step": 1,
    "horizontal_min": 0,      # Minimum time shift in seconds
    "horizontal_max": 1,      # Maximum time shift in seconds
    "horizontal_step": 0.1,   # Time shift step size
    "bpm_offset_min": -10,    # Minimum BPM offset
    "bpm_offset_max": 10,     # Maximum BPM offset
    "bpm_offset_step": 1      # BPM offset step size
}

# Output file names
INTERMEDIATE_MIDI_OUTPUT = "stage1_aligned_midi.mid"
FINAL_MIDI_OUTPUT = "stage2_optimized_midi.mid"
FIRST_STAGE_RESULTS = "first_search.csv"
SECOND_STAGE_RESULTS = "second_search.csv"

# Search metrics
FIRST_STAGE_METRIC = "cosine"    # Cosine similarity for first stage
SECOND_STAGE_METRIC = "mae"      # Mean Absolute Error for second stage

# Binarization settings
FIRST_STAGE_BINARIZE = False
SECOND_STAGE_BINARIZE = False

def load_reference_data(
    file_path: str, 
    trim_threshold: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process reference pitch contour data.
    
    Args:
        file_path: Path to the reference pitch contour CSV file
        trim_threshold: Fraction of total duration to keep (0-1)
        
    Returns:
        Tuple of (contour_values, time_stamps)
    """
    contour, times = load_reference_contour(file_path)
    return trim_contour_data(times, contour, times[-1] * trim_threshold)

def load_and_process_midi(
    file_path: str
) -> Tuple[float, float, List[Dict]]:
    """
    Load and process MIDI file.
    
    Args:
        file_path: Path to the MIDI file
        
    Returns:
        Tuple of (bpm, tick_rate, note_events_in_seconds)
    """
    mid = mido.MidiFile(file_path)
    bpm = get_midi_tempo(mid)
    tick_rate = get_midi_ticks_per_second(mid)
    events_ticks = convert_midi_to_note_intervals(mid)
    events_sec = convert_ticks_to_seconds(events_ticks, tick_rate, bpm)
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
) -> Dict[str, np.ndarray]:
    """
    Generate parameter ranges for the search algorithm.
    
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
        Dictionary containing parameter search ranges:
        - 'vertical': Array of pitch shift values in semitones
        - 'horizontal': Array of time shift values in seconds
        - 'scaling_factor': Array of BPM scaling factors
    """
    # Generate parameter ranges
    vertical_range = np.arange(vertical_min, vertical_max + vertical_step, vertical_step)
    horizontal_range = np.arange(horizontal_min, horizontal_max + horizontal_step, horizontal_step)
    
    # Generate valid BPM scaling factors
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

def run_search(
    strategy: str,
    reference_contour: np.ndarray,
    time_stamps: np.ndarray,
    note_events: List[Dict],
    search_space: Dict[str, np.ndarray],
    similarity_metric: str = "cosine",
    binarize: bool = False,
    maximize: bool = True
) -> Tuple[Tuple[float, float, float], float, List]:
    """
    Execute the search algorithm to find optimal alignment parameters.
    
    Args:
        strategy: Search strategy ('brute_force' or 'random')
        reference_contour: Array of reference pitch values
        time_stamps: Array of time stamps corresponding to pitch values
        note_events: List of MIDI note events
        search_space: Dictionary of parameter ranges
        similarity_metric: Metric to optimize (cosine or mae)
        binarize: Whether to binarize the data
        maximize: Whether to maximize the similarity metric
        
    Returns:
        Tuple of (best_parameters, best_similarity_score, search_record)
    """
    search_record = []
    
    if strategy == 'brute_force':
        best_params, best_similarity = brute_force_search(
            reference_contour,
            time_stamps,
            note_events,
            search_space,
            search_record,
            metric=similarity_metric,
            binarize=binarize,
            maximize=maximize
        )
    else:  # Default to random search
        best_params, best_similarity = random_search(
            reference_contour,
            time_stamps,
            note_events,
            search_space,
            search_record,
            n_iter=10,
            metric=similarity_metric,
            binarise=binarize
        )
    
    return best_params, best_similarity, search_record

def save_optimized_midi(
    parameters: Tuple[float, float, float],
    note_events: List[Dict],
    tick_rate: float,
    original_bpm: float,
    output_file: str
) -> None:
    """
    Apply transformations to MIDI note events and save the optimized MIDI file.
    
    Args:
        parameters: Tuple of (vertical_shift, horizontal_shift, scaling_factor)
        note_events: List of MIDI note event intervals
        tick_rate: MIDI ticks per second
        original_bpm: Original BPM of the MIDI file
        output_file: Path to save the optimized MIDI file
    """
    vertical_shift, horizontal_shift, scaling_factor = parameters
    
    # Apply transformations
    transformed_events = transform_note_intervals(
        note_events,
        vertical_shift,
        horizontal_shift,
        scaling_factor
    )
    
    # Convert back to ticks and create MIDI file
    transformed_ticks = convert_seconds_to_ticks(transformed_events, tick_rate, original_bpm)
    midi_file = create_midi_from_note_intervals(transformed_ticks, original_bpm, tick_rate)
    midi_file.save(output_file)

def find_best_parameters(
    search_space: Dict[str, np.ndarray],
    reference_contour: np.ndarray,
    note_events: List[Dict],
    time_stamps: np.ndarray,
    similarity_metric: str = "cosine",
    binarize: bool = False
) -> Tuple[Tuple[float, float, float], List]:
    """
    Find the best alignment parameters using brute force search.
    
    Args:
        search_space: Dictionary of parameter ranges
        reference_contour: Array of reference pitch values
        note_events: List of MIDI note events
        time_stamps: Array of time stamps
        similarity_metric: Metric to optimize
        binarize: Whether to binarize the data
        
    Returns:
        Tuple of (best_parameters, search_record)
    """
    # Determine whether to maximize or minimize the metric
    maximize = similarity_metric in ["cosine"]
    
    # Run brute force search
    best_params, best_similarity, search_record = run_search(
        'brute_force',
        reference_contour,
        time_stamps,
        note_events,
        search_space,
        similarity_metric=similarity_metric,
        binarize=binarize,
        maximize=maximize
    )
    
    # Print results
    print(f"\nSearch Results:")
    print(f"- Metric: {similarity_metric}")
    print(f"- Best Parameters: {best_params}")
    print(f"- Best Score: {best_similarity}")
    
    return best_params, search_record

def main():
    """
    Main function to run the alignment process.
    
    This function:
    1. Loads reference pitch contour and MIDI data
    2. Performs two-stage alignment:
       - First stage: Horizontal alignment and BPM scaling
       - Second stage: Vertical alignment (pitch shift)
    3. Saves results at each stage
    """
    print("Starting alignment process...")
    
    # Load reference data
    print("\nLoading reference data...")
    reference_contour, time_stamps = load_reference_data(
        f"{INPUT_DIRECTORY}/{REFERENCE_PITCH_FILE}", 
        TRIM_THRESHOLD
    )
    
    # Load and process MIDI
    print("\nProcessing MIDI file...")
    midi_bpm, tick_rate, note_events = load_and_process_midi(
        f"{INPUT_DIRECTORY}/{MIDI_INPUT_FILE}"
    )
    
    # Define search spaces
    print("\nSetting up search spaces...")
    vertical_search = define_search_space(midi_bpm, **VERTICAL_SEARCH_CONFIG)
    horizontal_bpm_search = define_search_space(midi_bpm, **HORIZONTAL_BPM_SEARCH_CONFIG)
    
    # First stage: Horizontal alignment and BPM scaling
    print("\nStarting first stage (horizontal alignment)...")
    intermediate_params, first_stage_record = find_best_parameters(
        horizontal_bpm_search,
        reference_contour,
        note_events,
        time_stamps,
        similarity_metric=FIRST_STAGE_METRIC,
        binarize=FIRST_STAGE_BINARIZE
    )
    
    # Save intermediate results
    print("\nSaving intermediate results...")
    save_optimized_midi(
        intermediate_params,
        note_events,
        tick_rate,
        midi_bpm / intermediate_params[2],  # Adjust BPM by scaling factor
        f"{OUTPUT_DIRECTORY}/{INTERMEDIATE_MIDI_OUTPUT}"
    )
    convert_search_record_to_dataframe(first_stage_record).to_csv(
        f"{OUTPUT_DIRECTORY}/{FIRST_STAGE_RESULTS}",
        index=False
    )
    
    # Apply first stage transformations
    transformed_note_events = transform_note_intervals(
        note_events,
        *intermediate_params
    )
    
    # Second stage: Vertical alignment
    print("\nStarting second stage (vertical alignment)...")
    final_params, second_stage_record = find_best_parameters(
        vertical_search,
        reference_contour,
        transformed_note_events,
        time_stamps,
        similarity_metric=SECOND_STAGE_METRIC,
        binarize=SECOND_STAGE_BINARIZE
    )
    
    # Save final results
    print("\nSaving final results...")
    save_optimized_midi(
        final_params,
        transformed_note_events,
        tick_rate,
        midi_bpm,
        f"{OUTPUT_DIRECTORY}/{FINAL_MIDI_OUTPUT}"
    )
    convert_search_record_to_dataframe(second_stage_record).to_csv(
        f"{OUTPUT_DIRECTORY}/{SECOND_STAGE_RESULTS}",
        index=False
    )
    
    print("\nAlignment process completed!")
    print(f"Final optimized parameters: {final_params}")
    print("- Vertical displacement: {:.2f} semitones".format(final_params[0]))
    print("- Horizontal displacement: {:.2f} seconds".format(final_params[1]))
    print("- Scaling factor: {:.4f}".format(final_params[2]))

if __name__ == "__main__":
    main()
