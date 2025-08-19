#!/usr/bin/env python
import torch
import sys
from pathlib import Path

def fix_checkpoint(input_path, output_path=None):
    """Fix a checkpoint file from a compiled model by removing _orig_mod prefixes"""
    print(f"Loading checkpoint from {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # Check if this is a compiled model checkpoint
    if not any(key.startswith("model._orig_mod") for key in checkpoint["state_dict"]):
        print("This checkpoint does not appear to be from a compiled model (no 'model._orig_mod' prefixes found).")
        print("No changes were made.")
        return
    
    # Create a new state dict with the '_orig_mod' part removed
    print("Converting model._orig_mod prefixes to model...")
    new_state_dict = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("model._orig_mod"):
            # Replace "model._orig_mod." with "model."
            new_key = key.replace("model._orig_mod.", "model.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Replace the original state dict with the corrected one
    checkpoint["state_dict"] = new_state_dict
    
    # Determine output path
    if output_path is None:
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
    
    # Save the fixed checkpoint
    print(f"Saving fixed checkpoint to {output_path}")
    torch.save(checkpoint, output_path)
    print("Done!")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_checkpoint.py <checkpoint_path> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    fix_checkpoint(input_path, output_path)