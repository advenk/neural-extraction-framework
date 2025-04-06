import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mlx.core as mx
import numpy as np
import argparse
from pathlib import Path
import gc
from tqdm import tqdm


def convert_tensor_chunk(tensor, chunk_size=100000):
    """Convert PyTorch tensor to MLX array in chunks to save memory."""
    if tensor.numel() <= chunk_size:
        return mx.array(tensor.numpy())
    
    # Convert to numpy first to avoid GPU operations
    tensor_np = tensor.numpy()
    chunks = []
    for i in range(0, tensor.numel(), chunk_size):
        end = min(i + chunk_size, tensor.numel())
        chunk = tensor_np.reshape(-1)[i:end]
        chunks.append(mx.array(chunk))
        del chunk
        gc.collect()
    
    return mx.concatenate(chunks).reshape(tensor.shape)


def save_weights_chunked(weights_dict, output_path, chunk_size=100):
    """Save weights dictionary in chunks to avoid memory issues."""
    # Get all keys and sort them for consistency
    all_keys = sorted(weights_dict.keys())
    num_chunks = (len(all_keys) + chunk_size - 1) // chunk_size
    
    # Save weights in chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(all_keys))
        chunk_keys = all_keys[start_idx:end_idx]
        
        chunk_dict = {k: weights_dict[k] for k in chunk_keys}
        chunk_path = str(output_path).replace('.npz', f'_chunk_{i}.npz')
        mx.savez(chunk_path, **chunk_dict)
        
        # Clear memory
        del chunk_dict
        gc.collect()


def convert_model(args):
    print(f"Loading model from {args.hf_path}...")
    
    # Load model in CPU mode with no offloading
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder=None,
        offload_state_dict=False,
        device_map=None
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path)

    # Create output directory
    output_dir = Path(args.mlx_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert state dict in chunks
    print("Converting model weights...")
    
    # Process in smaller groups
    layer_groups = []
    current_group = []
    current_size = 0
    max_group_size = 256 * 1024 * 1024  # 256MB per group
    
    for name, param in model.state_dict().items():
        param_size = param.numel() * param.element_size()
        if current_size + param_size > max_group_size and current_group:
            layer_groups.append(current_group)
            current_group = []
            current_size = 0
        current_group.append(name)
        current_size += param_size
    if current_group:
        layer_groups.append(current_group)

    # Process each group
    final_state_dict = {}
    for group_idx, group in enumerate(layer_groups):
        print(f"\nProcessing group {group_idx + 1}/{len(layer_groups)}")
        group_dict = {}
        for name in tqdm(group, desc=f"Group {group_idx + 1} parameters"):
            try:
                param = model.state_dict()[name]
                if param.device.type == 'meta':
                    param.data = param.data.to('cpu')
                group_dict[name] = convert_tensor_chunk(param.cpu(), chunk_size=100000)
                del param
                gc.collect()
            except Exception as e:
                print(f"Error converting parameter {name}: {e}")
                continue

        # Save intermediate state
        if group_idx < len(layer_groups) - 1:
            save_weights_chunked(group_dict, output_dir / f"weights_part_{group_idx}.npz")
        else:
            # For the last group, merge with final state dict
            final_state_dict.update(group_dict)
        
        del group_dict
        gc.collect()

    # Load and merge intermediate files
    print("\nMerging weights...")
    for i in range(len(layer_groups) - 1):
        base_path = output_dir / f"weights_part_{i}"
        chunk_idx = 0
        while True:
            chunk_path = Path(str(base_path).replace('.npz', f'_chunk_{chunk_idx}.npz'))
            if not chunk_path.exists():
                break
                
            try:
                intermediate_dict = mx.load(str(chunk_path))
                final_state_dict.update(intermediate_dict)
                del intermediate_dict
                gc.collect()
                chunk_path.unlink()
                chunk_idx += 1
            except Exception as e:
                print(f"Error loading chunk {chunk_idx} from group {i}: {e}")
                break

    # Save final weights in chunks
    print("Saving final weights...")
    save_weights_chunked(final_state_dict, output_dir / "weights.npz")
    del final_state_dict
    gc.collect()

    # Save config
    config = model.config.to_dict()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model converted and saved to {args.mlx_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to MLX format")
    parser.add_argument("--hf_path", type=str, required=True,
                      help="Path or name of the Hugging Face model")
    parser.add_argument("--mlx_path", type=str, required=True,
                      help="Output path for the MLX model")

    args = parser.parse_args()
    convert_model(args)


if __name__ == "__main__":
    main()
