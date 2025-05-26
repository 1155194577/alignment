import torch
import numpy as np
import pandas as pd
# Detect available acceleration
device = torch.device('mps' if torch.backends.mps.is_available() else 
                     'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
from midi_util import convert_midi_to_events, get_midi_bpm, normalize_midi_event, convert_events_to_midi_function, convert_event_to_midi

def load_reference_contour(file_path):
    """Load and convert reference contour to GPU tensor"""
    df = pd.read_csv(file_path)
    assert 'time' in df.columns, "CSV missing 'time' column"
    assert 'value' in df.columns, "CSV missing 'value' column"
    # Convert to float32 for MPS compatibility
    time = torch.from_numpy(df['time'].to_numpy().astype(np.float32) * 1000).to(device)
    pitches = torch.from_numpy(df['value'].to_numpy().astype(np.float32)).to(device)
    assert len(time) == len(pitches), "Time and pitch arrays must be the same length"
    return pitches, time

def transform_midi_events(midi_events, verticals, horizontals, bpms):
    """Batch transform MIDI events using tensor operations"""
    assert isinstance(midi_events, list) and len(midi_events) > 0, "midi_events must be a non-empty list"
    for e in midi_events:
        assert 'note' in e and 'start_time' in e and 'end_time' in e, "Each midi_event must have 'note', 'start_time', and 'end_time'"
    # Convert events to tensors
    notes = torch.tensor([e['note'] for e in midi_events], device=device)
    starts = torch.tensor([e['start_time'] for e in midi_events], device=device)
    ends = torch.tensor([e['end_time'] for e in midi_events], device=device)
    
    # Vectorized transformations with broadcasting (float32)
    assert verticals is not None and horizontals is not None and bpms is not None, "verticals, horizontals, and bpms must not be None"
    new_notes = (notes + verticals.view(-1, 1, 1)).float()
    scaled_starts = (starts * bpms.view(-1, 1, 1) + horizontals.view(-1, 1, 1)).float()
    scaled_ends = (ends * bpms.view(-1, 1, 1) + horizontals.view(-1, 1, 1)).float()
    
    return new_notes, scaled_starts, scaled_ends

def compute_similarity(reference, candidates):
    """GPU-accelerated similarity calculation"""
    assert reference is not None and candidates is not None, "reference and candidates must not be None"
    ref_norm = torch.norm(reference.float())
    cand_norms = torch.norm(candidates.float(), dim=1)
    dots = candidates.float() @ reference.float()
    assert ref_norm > 0, "Reference norm must be greater than zero"
    return dots / (ref_norm * cand_norms + 1e-8)

def brute_force_search(reference_contour, time, midi_events, search_space):
    """GPU-optimized parameter search"""
    assert isinstance(search_space, dict), "search_space must be a dictionary"
    assert 'vertical' in search_space and 'horizontal' in search_space and 'beat_per_minutes' in search_space, "search_space missing required keys"
    # Create parameter grid with float32
    verticals = torch.tensor(search_space['vertical'], device=device).float()
    horizontals = torch.tensor(search_space['horizontal'], device=device).float()
    bpms = torch.tensor(search_space['beat_per_minutes'], device=device).float()
    assert verticals.numel() > 0 and horizontals.numel() > 0 and bpms.numel() > 0, "Parameter grids must not be empty"
    
    v, h, b = torch.meshgrid(verticals, horizontals, bpms, indexing='ij')
    params = torch.stack((v.flatten(), h.flatten(), b.flatten()), dim=1)
    assert params.shape[0] > 0, "No parameter combinations generated"
    
    # Batch transform all parameters
    all_notes, all_starts, all_ends = transform_midi_events(midi_events, params[:,0], params[:,1], params[:,2])
    
    # Ensure correct shapes for broadcasting
    # all_notes: [P, N_notes], all_starts: [P, N_notes], all_ends: [P, N_notes]
    all_notes = all_notes  # [P, N_notes]
    all_starts = all_starts  # [P, N_notes]
    all_ends = all_ends      # [P, N_notes]
    time_tensor = time.to(device).float()  # [N_time]

    # Batch processing to avoid OOM
    batch_size = 128  # Adjust based on your GPU/CPU memory
    num_params = params.shape[0]
    best_score = -float('inf')
    best_param = None

    for i in range(0, num_params, batch_size):
        batch_slice = slice(i, min(i + batch_size, num_params))
        batch_notes = all_notes[batch_slice]
        batch_starts = all_starts[batch_slice]
        batch_ends = all_ends[batch_slice]

        # Reshape for broadcasting: [B, N_notes, 1] and [1, 1, N_time]
        batch_starts = batch_starts.unsqueeze(-1)  # [B, N_notes, 1]
        batch_ends = batch_ends.unsqueeze(-1)      # [B, N_notes, 1]
        time_tensor_b = time_tensor.view(1, 1, -1)  # [1, 1, N_time]

        active = (time_tensor_b >= batch_starts) & (time_tensor_b <= batch_ends)  # [B, N_notes, N_time]
        # Create contours matrix
        weighted_notes = batch_notes.unsqueeze(-1) * active.float()  # [B, N_notes, N_time]
        contours = weighted_notes.sum(dim=1)  # [B, N_time]

        similarities = compute_similarity(reference_contour.float(), contours)
        max_sim, max_idx = torch.max(similarities, dim=0)
        if max_sim.item() > best_score:
            best_score = max_sim.item()
            best_param = params[i + max_idx].cpu().numpy()

    return tuple(best_param), best_score
    
    # return tuple(params[best_idx].cpu().numpy()), similarities[best_idx].item()

if __name__ == "__main__":
    reference_contour, time = load_reference_contour("signal_a_contour_tony.csv")
    
    midi_events = convert_midi_to_events("signal_b.mid")
    midi_bpm = get_midi_bpm("signal_b.mid")
    normalized_events = normalize_midi_event(midi_events, midi_bpm)
    
    search_space = {
        'vertical': [0],
        'horizontal': np.arange(0, 5, 0.5).astype(np.float32),
        'beat_per_minutes': np.arange(midi_bpm-20, midi_bpm+20.5, 0.5).astype(np.float32)
    }
    
    best_params, best_score = brute_force_search(reference_contour, time, normalized_events, search_space)
    print(f"Best parameters: {best_params} with similarity {best_score}")
    
    # Save optimized MIDI
    transformed_events = transform_midi_events(normalized_events, *best_params)
    convert_event_to_midi(transformed_events, "optimized_signal_b.mid")
