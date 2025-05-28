import mido
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os # For os.path.basename

def convert_midi_to_events(file_path):
    """
    Processes a MIDI file and returns a list of note events with
    'start_time', 'end_time', 'note', and 'velocity'.
    """
    midi = mido.MidiFile(file_path)
    note_on_dict = {}
    events = []
    for track in midi.tracks:
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_on_dict[msg.note] = {
                    'start_time': time,
                    'note': msg.note,
                    'velocity': msg.velocity
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                note = msg.note
                if note in note_on_dict:
                    note_event = note_on_dict.pop(note)
                    note_event['end_time'] = time
                    events.append(note_event)
    print(f"Processed {len(events)} note events from {file_path}")
    return events

def convert_event_to_midi(events, file_path, tempo=500000):
    """
    Converts a list of MIDI events into a MIDI file and saves it to the specified path.
    Each event is expected to be a dictionary with 'start_time', 'end_time', 'note', and 'velocity'.
    Optionally sets the tempo (default 500000 microseconds per beat, i.e., 120 BPM).
    """
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Set tempo at the beginning
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

    # Sort events by start_time for correct delta times
    events_sorted = sorted(events, key=lambda e: e['start_time'])
    last_time = 0

    for event in events_sorted:
        start_time = int(event['start_time'])
        end_time = int(event['end_time'])
        note = event['note']
        velocity = event['velocity']

        # Delta time from last event
        delta_start = start_time - last_time
        track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta_start))
        track.append(mido.Message('note_off', note=note, velocity=velocity, time=end_time - start_time))
        last_time = end_time

    midi.save(file_path)
    print(f"Saved MIDI file with {len(events)} events to {file_path}")
    
def get_midi_bpm(file_path):
        """
        Returns the tempo (BPM) of the first tempo event in the MIDI file.
        If no tempo event is found, returns the default MIDI tempo (120 BPM).
        """
        midi = mido.MidiFile(file_path)
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    bpm = mido.tempo2bpm(msg.tempo)
                    return bpm
        return 120  # Default MIDI tempo

def get_midi_key_signature(file_path): 
    """
    Returns the key signature of the first key signature event in the MIDI file.
    If no key signature event is found, returns 'C major'.
    """
    midi = mido.MidiFile(file_path)
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'key_signature':
                return msg.key
    return 'C major'  # Default key signature

def convert_events_to_midi_function(events): 
    def midi_function(t):
        """
        input : time in milliseconds
        output : list of midi-note numbers active at that time
        """
        notes = []
        for event in events:
            if event['start_time'] <= t <= event['end_time']:
                notes.append(event['note'])
        return notes
    return midi_function 

def convert_midi_to_midi_function(file_path):
    """
    Converts a MIDI file to a function that returns active notes at a given time.
    """
    events = convert_midi_to_events(file_path)
    return convert_events_to_midi_function(events) 

# deprecated
# def normalize_midi_event(events, bpm):
#     """
#     Normalizes the MIDI events based on the BPM.
#     """
#     normalized_events = []
#     for event in events:
#         normalized_event = event.copy()
#         normalized_event['start_time'] = event['start_time'] * (bpm)
#         normalized_event['end_time'] = event['end_time'] * (bpm)
#         normalized_events.append(normalized_event)
#     return normalized_events