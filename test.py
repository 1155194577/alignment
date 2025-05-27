from midi_util import convert_midi_to_events, get_midi_bpm, normalize_midi_event, convert_event_to_midi
from brute_force import transform_midi_events 
import mido
midi_file_path = "signal_b.mid"
midi_events = convert_midi_to_events(midi_file_path) 
print(midi_events[0])
midi_bpm = get_midi_bpm(midi_file_path)
print(midi_bpm)
normalized_midi_events = normalize_midi_event(midi_events,midi_bpm)
print(normalized_midi_events[0])
transformed_midi_events = transform_midi_events(normalized_midi_events, 0, 0, 105)  # Example parameters 
print(transformed_midi_events[0])
# bpm = mido.tempo2bpm(msg.tempo)
tempo = mido.bpm2tempo(500)  # Convert BPM to MIDI tempo format 
convert_event_to_midi(transformed_midi_events, "transformed_signal_b.mid",tempo)  # Save the transformed MIDI events to a file