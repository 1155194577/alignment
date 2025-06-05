from search_algorithm import brute_force_search, random_search
from midi_processing import convert_midi_to_note_event_intervals,convert_tick_to_seconds,get_midi_bpm,get_midi_tick_per_second,convert_note_event_intervals_to_midi_function,convert_midi_function_to_midi_contour,convert_midi_contour_to_pitch_contour,transform_note_event_interval,convert_note_event_intervals_to_absolute_events,convert_absolute_events_to_relative,relative_events_to_midi,convert_seconds_to_tick,convert_note_event_intervals_to_mid
from audio_processing import load_reference_contour,trim_times_below_threshold
import mido
from util import convert_search_record_to_dataframe
import numpy as np
def main(): 
    path_to_reference_contour = "./john/ep_030_pitch.csv"
    reference_contour, times = load_reference_contour(path_to_reference_contour) ## Load the reference contour from a CSV file
    reference_contour , times = trim_times_below_threshold(times, reference_contour, 20)  ## Trim the reference contour to only include entries where times are below 3 seconds

    path_to_midi_file = "./john/ep_030.mid"
    mid = mido.MidiFile(path_to_midi_file)

    midi_bpm = get_midi_bpm(mid) 
    midi_tick_per_second = get_midi_tick_per_second(mid)
    note_event_intervals_ticks =  convert_midi_to_note_event_intervals(mid) ## in ticks 
    note_event_intervals_seconds = convert_tick_to_seconds(note_event_intervals_ticks, midi_tick_per_second, midi_bpm) ## in seconds
   
    horizontal_range = np.arange(0, 2, 0.1)
    midi_bpm_range = [midi_bpm + offset for offset in np.arange(-10, 10, 1)]
    scaling_factor_range1 = np.array([midi_bpm/x for x in midi_bpm_range])
    scaling_factor_range2 = np.array([x/midi_bpm for x in midi_bpm_range])
    scaling_factor_range = np.unique(np.concatenate([scaling_factor_range1, scaling_factor_range2]))
    # scaling_factor_range = np.arange(0.1, 5, 0.2)  
    search_record = [] 
    search_space = {
        'vertical': [0],  # Vertical displacements
        'horizontal': horizontal_range, 
        'scaling_factor': scaling_factor_range 
    }

    best_parameter, best_similarity = brute_force_search(reference_contour,times, note_event_intervals_seconds, search_space, search_record)
    # best_parameter, best_similarity = random_search(reference_contour, times, note_event_intervals_seconds, search_space, search_record, n_iter=10)  # Perform the search for the best parameters
    vertical_displacement, horizontal_displacement, scaling_factor = best_parameter
    print("Best parameters found:", best_parameter)  # Print the best parameters found
    print("Best similarity:", best_similarity)  # Print the best similarity score
    print("original bpm ,new bpm", midi_bpm, midi_bpm / scaling_factor)  # Print the new BPM after applying the best scaling factor

    # print("Best similarity:", best_similarity)
    convert_search_record_to_dataframe(search_record).to_csv("search_results.csv", index=False)  # Save the search record to a CSV file
    optimized_note_event_interval = transform_note_event_interval(note_event_intervals_seconds, vertical_displacement,horizontal_displacement,scaling_factor)  # Apply the best parameters to the note event intervals
    tick_optimized_note_event_interval = convert_seconds_to_tick(optimized_note_event_interval, midi_tick_per_second, midi_bpm)  # Convert the optimized intervals back to ticks
    # print("Optimized note event intervals in ticks:", tick_optimized_note_event_interval)  # Print the optimized intervals in ticks

    mid = convert_note_event_intervals_to_mid(tick_optimized_note_event_interval,bpm=midi_bpm,ticks_per_beat=midi_tick_per_second)  # Convert the optimized note event intervals back to a MIDI file
    # print(mid)
    mid.save("./johnny_simple_midi/Johnny_simpleMIDI_optimized.mid")
if __name__ == "__main__":
    main()

