import mido
import numpy as np
def midi_to_relative_events(mid):
    """
    Converts MIDI tracks into relative time events
    using a pre-loaded MidiFile object
    """
    events = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                events.append({
                    'type': msg.type,
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'time': msg.time  # relative delta time
                })
    return events

def convert_relative_events_to_absolute(events):
    """
    Converts relative MIDI events to absolute time.
    """
    absolute_events = []
    current_time = 0
    for event in events:
        current_time += event['time']
        absolute_events.append({
            'type': event['type'],
            'note': event['note'],
            'velocity': event['velocity'],
            'time': current_time
        })
    return absolute_events

def convert_absolute_events_to_note_event_intervals(events):
    """
    Converts absolute MIDI events to note event intervals
    using a dictionary for active notes.
    """
    note_events = []
    active_notes = {}
    for event in events:
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
                note_events.append(note_event)
    return note_events

def convert_tick_to_seconds(events, ticks_per_beat, bpm):
    """
    Converts MIDI events from ticks to seconds
    using mido's built-in tick2second.
    """
    tempo = mido.bpm2tempo(bpm)
    for event in events:
        event["start_time"] = mido.tick2second(event['start_time'], ticks_per_beat, tempo)
        event["end_time"] = mido.tick2second(event['end_time'], ticks_per_beat, tempo)
    return events

def convert_note_event_intervals_to_midi_function(events):
    """
    Converts note event intervals to a MIDI function
    that returns active notes at a given time.
    """
    return lambda t: [event['note'] for event in events if event['start_time'] <= t <= event['end_time']]

def convert_midi_function_to_midi_contour(midi_function, times):
    """
    Given a midi_function and an iterable of time values,
    returns a list of active notes for each time value.
    """
    midi_contour = [] 
    for t in times: 
        note_number = midi_function(t) 
        if len(note_number) > 0: 
            midi_contour.append(note_number[0]) 
        else: 
            midi_contour.append(0) 
    return np.array(midi_contour)

def convert_midi_contour_to_pitch_contour(midi_contour):
    """
    Converts a MIDI contour (numpy array) to a pitch contour (numpy array).
    """
    midi_contour = np.asarray(midi_contour)
    pitch_contour = np.zeros_like(midi_contour, dtype=float)
    mask = midi_contour > 0
    pitch_contour[mask] = 440.0 * (2 ** ((midi_contour[mask] - 69) / 12.0))
    return pitch_contour

def convert_pitch_contour_to_midi_contour(pitch_contour):
    """
    Converts a pitch contour to a MIDI contour.
    Inverse of convert_midi_contour_to_pitch_contour.
    """
    pitch_to_note = lambda freq: 69 + 12 * np.log2(freq / 440.0)
    return np.array([round(pitch_to_note(freq)) if freq > 0 else 0 for freq in pitch_contour])

def get_midi_bpm(mid):
    """
    Returns the tempo (BPM) of the first tempo event
    in the MidiFile object.
    If no tempo event found, returns 120 BPM.
    """
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)
    return 120

def get_midi_tick_per_second(mid):
    """
    Returns the ticks per beat of the MidiFile object.
    """
    return mid.ticks_per_beat

def relative_events_to_midi(events, bpm, ticks_per_second):
    """
    Converts relative events back to a MIDI file.
    Inverse of midi_to_relative_events.
    Accepts ticks_per_second instead of ticks_per_beat.
    """
    # MIDI standard uses ticks_per_beat, so we need to convert ticks_per_second to ticks_per_beat
    # ticks_per_beat = ticks_per_second * 60 / bpm
    ticks_per_beat = int(round(ticks_per_second * 60 / bpm))
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Add tempo meta message at the start
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    
    for event in events:
        msg = mido.Message(event['type'], note=event['note'], 
                          velocity=event['velocity'], time=event['time'])
        track.append(msg)
    
    return mid

def convert_absolute_events_to_relative(absolute_events):
    """
    Converts absolute events to relative events.
    Inverse of convert_relative_events_to_absolute.
    """
    relative_events = []
    prev_time = 0
    
    for event in absolute_events:
        relative_time = event['time'] - prev_time
        relative_events.append({
            'type': event['type'],
            'note': event['note'],
            'velocity': event['velocity'],
            'time': relative_time
        })
        prev_time = event['time']
    
    return relative_events

def convert_note_event_intervals_to_absolute_events(note_events):
    """
    Converts note event intervals back to absolute events.
    Inverse of convert_absolute_events_to_note_event_intervals.
    """
    absolute_events = []
    
    for note in note_events:
        # Create note_on event
        absolute_events.append({
            'type': 'note_on',
            'note': note['note'],
            'velocity': note['velocity'],
            'time': note['start_time']
        })
        # Create note_off event
        absolute_events.append({
            'type': 'note_off',
            'note': note['note'],
            'velocity': 0,
            'time': note['end_time']
        })
    
    # Sort events by time
    absolute_events.sort(key=lambda x: x['time'])
    return absolute_events

def convert_seconds_to_tick(events, ticks_per_beat, bpm):
    """
    Converts note event intervals from seconds back to ticks.
    Inverse of convert_tick_to_seconds.
    """
    tempo = mido.bpm2tempo(bpm)
    for event in events:
        event["start_time"] = mido.second2tick(event['start_time'], ticks_per_beat, tempo)
        event["end_time"] = mido.second2tick(event['end_time'], ticks_per_beat, tempo)
    return events

## public
def convert_midi_to_note_event_intervals(mid):
    """
    Converts a MIDI file to note event intervals.
    """
    relative_events = midi_to_relative_events(mid)
    absolute_events = convert_relative_events_to_absolute(relative_events)
    note_intervals = convert_absolute_events_to_note_event_intervals(absolute_events)
    
    return note_intervals

def convert_midi_to_midi_contour(mid, times):
    """
    Converts a MIDI file to a MIDI contour.
    """
    note_intervals = convert_midi_to_note_event_intervals(mid)
    note_intervals_with_seconds = convert_tick_to_seconds(note_intervals, get_midi_tick_per_second(mid), get_midi_bpm(mid))
    midi_function = convert_note_event_intervals_to_midi_function(note_intervals_with_seconds)
    return convert_midi_function_to_midi_contour(midi_function, times) 

def transform_note_event_interval(note_events, vertical_displacement, horizontal_displacement, scaling_factor):
        """
        Apply pitch shift, timing shift, and tempo scaling to note event intervals.
        Args:
            note_events (list of dict): Each dict has 'start_time', 'end_time', 'note', 'velocity'.
            vertical_displacement (float): Semitone shift for pitch.
            horizontal_displacement (float): Time shift in milliseconds or ticks.
            scaling_factor (float): Factor to scale timing (tempo adjustment).
        Returns:
            list of dict: Transformed note event intervals.
        """
        transformed_note_event_intervals = []
        for event in note_events:
            new_event = event.copy()
            new_event['note'] = event['note'] + vertical_displacement
            new_event['start_time'] = (event['start_time'] * scaling_factor) + horizontal_displacement
            new_event['end_time'] = (event['end_time'] * scaling_factor) + horizontal_displacement
            transformed_note_event_intervals.append(new_event)
        return transformed_note_event_intervals

def convert_note_event_intervals_to_mid(note_events, bpm=120, ticks_per_beat=480):
            """
            Converts note event intervals to a MidiFile object.
            Args:
                note_events (list of dict): Each dict has 'start_time', 'end_time', 'note', 'velocity'.
                bpm (float): Tempo in beats per minute.
                ticks_per_beat (int): MIDI ticks per beat.
            Returns:
                mido.MidiFile: The resulting MIDI file.
            """
            # Convert note event intervals to absolute events
            absolute_events = convert_note_event_intervals_to_absolute_events(note_events)
            # Convert absolute events to relative events
            relative_events = convert_absolute_events_to_relative(absolute_events)
            # Create MidiFile
            mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            # Add tempo meta message at the start
            tempo = mido.bpm2tempo(bpm)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
            # Add note events
            for event in relative_events:
                msg = mido.Message(event['type'], note=event['note'],
                                   velocity=event['velocity'], time=event['time'])
                track.append(msg)
            return mid
