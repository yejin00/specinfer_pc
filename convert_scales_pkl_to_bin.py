#!/usr/bin/env python3
"""
Convert KVQuant PKL (activation min/max) to per-channel scales binary.

Supports two formats:
1) Top-level dict with keys 'min' and 'max' (arrays of length N)
2) Top-level dict of per-layer entries, each entry a dict with 'min'/'max'

Aggregation across layers: max over layers (safe for avoiding clipping).
If the per-channel vector length is divisible by 128, we auto-reduce to head_dim=128
by grouping (D // 128, 128) and taking mean across groups. This matches LLaMA-style
head_dim commonly being 128. If not divisible, we keep original length.

Usage:
    python convert_scales_pkl_to_bin.py /path/to/q_k_only.pkl /path/to/output/scales_k.bin
"""

import pickle
import numpy as np
import sys
import os
from typing import Optional, Tuple

try:
    import torch  # type: ignore
except Exception:
    torch = None

def _amax_from_minmax(mins, maxs):
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)
    if mins.shape != maxs.shape:
        raise ValueError("min and max shapes don't match")
    amax = np.maximum(np.abs(mins), np.abs(maxs))
    # Flatten to 1D channel vector (handles shapes like (1, 4096))
    return amax.reshape(-1)


def _reduce_to_head_dim(vec, head_dim=128):
    vec = np.asarray(vec, dtype=np.float32)
    D = vec.shape[0]
    if D % head_dim != 0:
        # cannot reduce cleanly; return as-is
        return vec
    groups = D // head_dim
    # reshape (groups, head_dim) and average across groups
    return vec.reshape(groups, head_dim).mean(axis=0)


def convert_pkl_to_scales(pkl_path, output_path, no_reduce=False, layer_index=None):
    """
    Convert pkl file with min/max to per-channel scales.
    
    Scale formula: scale[ch] = max(|min[ch]|, |max[ch]|) / 8
    """
    print(f"Loading {pkl_path}...")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Top-level keys: {list(data.keys())[:4]}{'...' if len(data) > 4 else ''}")

    # Helper: extract amax from a single layer entry in flexible formats
    def _extract_layer_amax(entry) -> Optional[np.ndarray]:
        # dict with explicit keys
        if isinstance(entry, dict):
            if 'min' in entry and 'max' in entry:
                return _amax_from_minmax(entry['min'], entry['max'])
            if 'amin' in entry and 'amax' in entry:
                return _amax_from_minmax(entry['amin'], entry['amax'])
            if 'minmax' in entry:
                mm = entry['minmax']
                # expect (2, N) or (N, 2) or list[2][N]
                arr = np.asarray(mm)
                if arr.ndim == 2 and 2 in arr.shape:
                    if arr.shape[0] == 2:
                        mins, maxs = arr[0], arr[1]
                    else:
                        mins, maxs = arr[:, 0], arr[:, 1]
                    return _amax_from_minmax(mins, maxs)
        # tuple/list of (min, max)
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            mins, maxs = entry[0], entry[1]
            return _amax_from_minmax(mins, maxs)
        # numpy array with last dim=2
        if isinstance(entry, np.ndarray):
            if entry.ndim == 2 and entry.shape[1] == 2:
                return _amax_from_minmax(entry[:, 0], entry[:, 1])
            if entry.ndim == 2 and entry.shape[0] == 2:
                return _amax_from_minmax(entry[0], entry[1])
        # torch tensor with last dim=2
        if torch is not None and isinstance(entry, torch.Tensor):
            arr = entry.detach().cpu().numpy()
            if arr.ndim == 2 and arr.shape[1] == 2:
                return _amax_from_minmax(arr[:, 0], arr[:, 1])
            if arr.ndim == 2 and arr.shape[0] == 2:
                return _amax_from_minmax(arr[0], arr[1])
        return None

    # Case 1: direct min/max at top-level
    if isinstance(data, dict) and ('min' in data and 'max' in data):
        amax = _amax_from_minmax(data['min'], data['max'])
        print("Detected format: single min/max at top-level")
    # Case 2: dict of layers with per-layer min/max (flexible)
    elif isinstance(data, dict):
        print("Detected format: dict of layers (flexible per-layer structures)")
        # If a specific layer is requested, use only that
        if layer_index is not None:
            preferred_key = f"model.layers.{layer_index}.self_attn.k_proj"
            if preferred_key in data:
                entry = data[preferred_key]
            else:
                # fallback: pick by sorted order index
                keys_sorted = sorted(data.keys())
                if 0 <= layer_index < len(keys_sorted):
                    entry = data[keys_sorted[layer_index]]
                else:
                    print(f"Error: layer_index {layer_index} out of range (0..{len(data)-1})")
                    return False
            amax = _extract_layer_amax(entry)
            if amax is None:
                print("Error: Could not parse per-layer min/max for the selected layer")
                return False
            amax = np.asarray(amax, dtype=np.float32).reshape(-1)
            print(f"Using only layer_index={layer_index}, channels={amax.shape[0]}")
        else:
            # Aggregate across all layers
            amax_list = []
            debug_first = True
            for k in sorted(data.keys()):
                entry = data[k]
                if debug_first:
                    print(f"DEBUG: First layer '{k}' entry type={type(entry)}")
                    if isinstance(entry, np.ndarray):
                        print(f"  ndarray shape={entry.shape}, dtype={entry.dtype}")
                    elif isinstance(entry, (list, tuple)):
                        print(f"  list/tuple len={len(entry)}, first elem type={type(entry[0]) if len(entry) > 0 else 'empty'}")
                        if len(entry) >= 2 and torch is not None:
                            if isinstance(entry[0], torch.Tensor):
                                print(f"  entry[0] (min) shape={entry[0].shape}")
                            if isinstance(entry[1], torch.Tensor):
                                print(f"  entry[1] (max) shape={entry[1].shape}")
                    elif isinstance(entry, dict):
                        print(f"  dict keys={list(entry.keys())}")
                    debug_first = False
                amax_layer = _extract_layer_amax(entry)
                if amax_layer is None:
                    continue
                amax_list.append(amax_layer)
            if not amax_list:
                print("Error: No valid per-layer min/max entries found")
                return False
            amax = np.max(np.stack(amax_list, axis=0), axis=0)
            amax = np.asarray(amax, dtype=np.float32).reshape(-1)
            print(f"Aggregated across {len(amax_list)} layers using max")
    else:
        print("Error: Unsupported PKL structure")
        return False

    n_channels = int(amax.shape[0])
    print(f"n_channels (before optional reduction): {n_channels}")

    # Optional reduction to head_dim=128 if applicable
    if not no_reduce:
        amax_reduced = _reduce_to_head_dim(amax, head_dim=128)
        if amax_reduced.shape[0] != amax.shape[0]:
            print(f"Reduced from model dim {amax.shape[0]} to head_dim {amax_reduced.shape[0]} via mean across groups")
            amax = amax_reduced
    n_channels = int(amax.shape[0])
    print(f"Final number of channels: {n_channels}")

    # Calculate per-channel scales: scale = amax / 8
    amax = np.asarray(amax, dtype=np.float32)
    scales = amax / 8.0
    
    # Handle zero scales
    zero_mask = (amax == 0)
    if np.any(zero_mask):
        print(f"Warning: {np.sum(zero_mask)} channels have zero range, setting scale to 1e-8")
        scales[zero_mask] = 1e-8
    
    # Statistics
    print(f"\nScale statistics:")
    print(f"  Min scale: {scales.min():.6f}")
    print(f"  Max scale: {scales.max():.6f}")
    print(f"  Mean scale: {scales.mean():.6f}")
    print(f"  Median scale: {np.median(scales):.6f}")
    
    # Save as float32 binary
    scales_f32 = scales.astype(np.float32)
    scales_f32.tofile(output_path)
    
    # Verify
    file_size = os.path.getsize(output_path)
    expected_size = n_channels * 4  # 4 bytes per float32
    
    print(f"\nSaved to {output_path}")
    print(f"File size: {file_size} bytes (expected {expected_size} bytes)")
    
    if file_size != expected_size:
        print("Error: File size mismatch!")
        return False
    
    # Verify by reading back
    verify = np.fromfile(output_path, dtype=np.float32)
    if np.allclose(verify, scales_f32):
        print("✓ Verification passed!")
    else:
        print("✗ Verification failed!")
        return False
    
    return True

if __name__ == "__main__":
    # Args: <input.pkl> <output.bin> [--no-reduce] [--layer N]
    if len(sys.argv) < 3:
        print("Usage: python convert_scales_pkl_to_bin.py <input.pkl> <output.bin> [--no-reduce] [--layer N]")
        print("\nExamples:")
        print("  python convert_scales_pkl_to_bin.py /home/yjkim00/KVQuant/q_k_only.pkl scales_k.bin --no-reduce")
        print("  python convert_scales_pkl_to_bin.py /home/yjkim00/KVQuant/q_k_only.pkl scales_k.bin --layer 0")
        sys.exit(1)

    pkl_path = sys.argv[1]
    output_path = sys.argv[2]

    # Parse optional flags
    no_reduce = False
    layer_index = None
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--no-reduce":
            no_reduce = True
            i += 1
        elif sys.argv[i] == "--layer" and i + 1 < len(sys.argv):
            try:
                layer_index = int(sys.argv[i+1])
            except ValueError:
                print("Error: --layer expects an integer")
                sys.exit(1)
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)

    if not os.path.exists(pkl_path):
        print(f"Error: Input file not found: {pkl_path}")
        sys.exit(1)
    
    success = convert_pkl_to_scales(pkl_path, output_path, no_reduce=no_reduce, layer_index=layer_index)
    
    if success:
        print("\n✓ Conversion completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Conversion failed!")
        sys.exit(1)
