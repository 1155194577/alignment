from typing import List, Dict
import mido
import numpy as np

def midi_to_relative_events(
    midi_file: mido.MidiFile
) -> List[Dict]:
    """
    Convert MIDI tracks into relative time events.
    
    Args:
        midi_file: Loaded MidiFile object containing MIDI tracks
        
    Returns:
        List of MIDI events with relative timing
    """
    events = []
    for track in midi_file.tracks:
        absolute_time = 0
        for message in track:
            absolute_time += message.time
            if message.type in ['note_on', 'note_off']:
                events.append({
                    'type': message.type,
                    'note': message.note,
                    'velocity': message.velocity,
                    'time': message.time  # relative delta time
                })
    return events

def convert_relative_to_absolute_events(
    relative_events: List[Dict]
) -> List[Dict]:
    """
    Convert relative MIDI events to absolute timing.
    
    Args:
        relative_events: List of MIDI events with relative timing
        
    Returns:
        List of MIDI events with absolute timing
    """
    absolute_events = []
    current_time = 0
    for event in relative_events:
        current_time += event['time']
        absolute_events.append({
            'type': event['type'],
            'note': event['note'],
            'velocity': event['velocity'],
            'time': current_time
        })
    return absolute_events

def convert_absolute_to_note_intervals(
    absolute_events: List[Dict]
) -> List[Dict]:
    """
    Convert absolute MIDI events to note event intervals.
    
    Args:
        absolute_events: List of MIDI events with absolute timing
        
    Returns:
        List of note event intervals with start and end times
    """
    note_intervals = []
    active_notes = {}
    for event in absolute_events:
        if event['type'] == 'note_on' and event['velocity'] > 0:
            active_notes[event['note']] = {
                'start_time': event['time'],
                'note': event['note'],
                'velocity': event['velocity']
            }
        elif (event['type'] == 'note_off') or (event['type'] == 'note_on' and event['velocity'] == 0):
            if event['note'] in active_notes:
                note_event = active_notes.pop(event['note'])
                note_event['end_time'] = event['time']
                note_intervals.append(note_event)
    return note_intervals

def convert_ticks_to_seconds(
    events: List[Dict],
    ticks_per_beat: int,
    bpm: float
) -> List[Dict]:
    """
    Convert MIDI events from ticks to seconds.
    
    Args:
        events: List of MIDI events
        ticks_per_beat: MIDI ticks per beat
        bpm: Beats per minute
        
    Returns:
        List of events with timing in seconds
    """
    tempo = mido.bpm2tempo(bpm)
    for event in events:
        event["start_time"] = mido.tick2second(event['start_time'], ticks_per_beat, tempo)
        event["end_time"] = mido.tick2second(event['end_time'], ticks_per_beat, tempo)
    return events

def create_midi_function(
    note_intervals: List[Dict]
) -> callable:
    """
    Create a function that returns active notes at a given time.
    
    Args:
        note_intervals: List of note event intervals
        
    Returns:
        Function that takes time and returns list of active notes
    """
    return lambda t: [event['note'] for event in note_intervals if event['start_time'] <= t <= event['end_time']]

def create_midi_contour(
    midi_function: callable,
    time_stamps: np.ndarray
) -> np.ndarray:
    """
    Generate MIDI contour from MIDI function and time values.
    
    Args:
        midi_function: Function returning active notes at given time
        time_stamps: Array of time values
        
    Returns:
        Array of MIDI note numbers (0 for no active note)
    """
    midi_contour = []
    for t in time_stamps:
        note_numbers = midi_function(t)
        if len(note_numbers) > 0:
            midi_contour.append(note_numbers[0])
        else:
            midi_contour.append(0)
    return np.array(midi_contour)

def convert_midi_to_pitch(
    midi_contour: np.ndarray
) -> np.ndarray:
    """
    Convert MIDI contour to pitch contour.
    
    Args:
        midi_contour: Array of MIDI note numbers
        
    Returns:
        Array of pitch values in Hz
    """
    midi_contour = np.asarray(midi_contour)
    pitch_contour = np.zeros_like(midi_contour, dtype=float)
    mask = midi_contour > 0
    pitch_contour[mask] = 440.0 * (2 ** ((midi_contour[mask] - 69) / 12.0))
    return pitch_contour

def convert_pitch_to_midi(
    pitch_contour: np.ndarray
) -> np.ndarray:
    """
    Convert pitch contour to MIDI contour.
    
    Args:
        pitch_contour: Array of pitch values in Hz
        
    Returns:
        Array of MIDI note numbers
    """
    pitch_contour = np.asarray(pitch_contour)
    midi_contour = np.zeros_like(pitch_contour, dtype=int)
    mask = pitch_contour > 0
    midi_contour[mask] = np.round(69 + 12 * np.log2(pitch_contour[mask] / 440.0))
    return midi_contour

def get_midi_tempo(
    midi_file: mido.MidiFile
) -> float:
    """
    Get the tempo (BPM) from the MIDI file.
    
    Args:
        midi_file: Loaded MidiFile object
        
    Returns:
        Tempo in BPM (120 if no tempo event found)
    """
    for message in midi_file:
        if message.type == 'set_tempo':
            return mido.tempo2bpm(message.tempo)
    return 120.0

def get_midi_ticks_per_second(
    midi_file: mido.MidiFile
) -> int:
    """
    Get the ticks per beat from the MIDI file.
    
    Args:
        midi_file: Loaded MidiFile object
        
    Returns:
        Ticks per beat
    """
    return midi_file.ticks_per_beat

def create_midi_from_relative_events(
    relative_events: List[Dict],
    bpm: float,
    ticks_per_second: int
) -> mido.MidiFile:
    """
    Create MIDI file from relative events.
    
    Args:
        relative_events: List of MIDI events with relative timing
        bpm: Tempo in BPM
        ticks_per_second: MIDI ticks per second
        
    Returns:
        MidiFile object
    """
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    
    for event in relative_events:
        track.append(mido.Message(
            event['type'],
            note=event['note'],
            velocity=event['velocity'],
            time=event['time']
        ))
    
    return midi_file

def convert_absolute_to_relative_events(
    absolute_events: List[Dict]
) -> List[Dict]:
    """
    Convert absolute events to relative events.
    
    Args:
        absolute_events: List of MIDI events with absolute timing
        
    Returns:
        List of MIDI events with relative timing
    """
    relative_events = []
    current_time = 0
    for event in absolute_events:
        relative_events.append({
            'type': event['type'],
            'note': event['note'],
            'velocity': event['velocity'],
            'time': event['time'] - current_time
        })
        current_time = event['time']
    return relative_events

def convert_note_intervals_to_absolute_events(
    note_intervals: List[Dict]
) -> List[Dict]:
    """
    Convert note event intervals back to absolute events.
    
    Args:
        note_intervals: List of note event intervals
        
    Returns:
        List of MIDI events with absolute timing
    """
    absolute_events = []
    for event in note_intervals:
        absolute_events.append({
            'type': 'note_on',
            'note': event['note'],
            'velocity': event['velocity'],
            'time': event['start_time']
        })
        absolute_events.append({
            'type': 'note_off',
            'note': event['note'],
            'velocity': 0,
            'time': event['end_time']
        })
    return absolute_events

def convert_seconds_to_ticks(
    events: List[Dict],
    ticks_per_beat: int,
    bpm: float
) -> List[Dict]:
    """
    Convert MIDI events from seconds to ticks.
    
    Args:
        events: List of MIDI events with timing in seconds
        ticks_per_beat: MIDI ticks per beat
        bpm: Tempo in BPM
        
    Returns:
        List of events with timing in ticks
    """
    tempo = mido.bpm2tempo(bpm)
    for event in events:
        event["start_time"] = mido.second2tick(event['start_time'], ticks_per_beat, tempo)
        event["end_time"] = mido.second2tick(event['end_time'], ticks_per_beat, tempo)
    return events

def convert_midi_to_note_intervals(
    midi_file: mido.MidiFile
) -> List[Dict]:
    """
    Convert MIDI file to note event intervals.
    
    Args:
        midi_file: Loaded MidiFile object
        
    Returns:
        List of note event intervals
    """
    relative_events = midi_to_relative_events(midi_file)
    absolute_events = convert_relative_to_absolute_events(relative_events)
    return convert_absolute_to_note_intervals(absolute_events)

def convert_midi_to_contour(
    midi_file: mido.MidiFile,
    time_stamps: np.ndarray
) -> np.ndarray:
    """
    Convert MIDI file to MIDI contour.
    
    Args:
        midi_file: Loaded MidiFile object
        time_stamps: Array of time values
        
    Returns:
        Array of MIDI note numbers
    """
    note_intervals = convert_midi_to_note_intervals(midi_file)
    midi_function = create_midi_function(note_intervals)
    return create_midi_contour(midi_function, time_stamps)

def transform_note_intervals(
    note_intervals: List[Dict],
    vertical_shift: float,
    horizontal_shift: float,
    scaling_factor: float
) -> List[Dict]:
    """
    Apply pitch shift, timing shift, and tempo scaling to note event intervals.
    
    Args:
        note_intervals: List of note event intervals
        vertical_shift: Pitch shift in semitones
        horizontal_shift: Time shift in seconds
        scaling_factor: Factor to scale timing (tempo adjustment)
        
    Returns:
        List of transformed note event intervals
    """
    transformed_events = []
    for event in note_intervals:
        # Apply pitch shift
        transformed_note = event['note'] + vertical_shift
        if transformed_note < 0:
            transformed_note = 0
        elif transformed_note > 127:
            transformed_note = 127
        
        # Apply timing shifts
        transformed_start = event['start_time'] * scaling_factor + horizontal_shift
        transformed_end = event['end_time'] * scaling_factor + horizontal_shift
        
        # Ensure valid timing
        if transformed_start < 0:
            transformed_start = 0
        if transformed_end < 0:
            transformed_end = 0
        
        transformed_events.append({
            'note': int(transformed_note),
            'velocity': event['velocity'],
            'start_time': transformed_start,
            'end_time': transformed_end
        })
    return transformed_events

def create_midi_from_note_intervals(
    note_intervals: List[Dict],
    bpm: float = 120,
    ticks_per_beat: int = 480
) -> mido.MidiFile:
    """
    Create MIDI file from note event intervals.
    
    Args:
        note_intervals: List of note event intervals
        bpm: Tempo in BPM
        ticks_per_beat: MIDI ticks per beat
        
    Returns:
        MidiFile object
    """
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    
    # Add tempo meta message
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    
    # Sort events by start time
    note_intervals.sort(key=lambda x: x['start_time'])
    
    # Add note_on and note_off messages
    for event in note_intervals:
        # Add note_on message
        track.append(mido.Message(
            'note_on',
            note=event['note'],
            velocity=event['velocity'],
            time=mido.second2tick(event['start_time'], ticks_per_beat, mido.bpm2tempo(bpm))
        ))
        
        # Add note_off message
        track.append(mido.Message(
            'note_off',
            note=event['note'],
            velocity=0,
            time=mido.second2tick(event['end_time'] - event['start_time'], ticks_per_beat, mido.bpm2tempo(bpm))
        ))
    
    return midi_file
