"""
Helper Script to convert Megatron shards to safetensor model files, compatible with Huggingface API

The main purpose is to be able to enable users who choose not to enable HF model saves during training, such as enable the `hf_save_interval` parameter, to
also be able to benefit from a way to create a HF safetensors model.

For Megatron, the input directory should resemble the following structure:
.
|-- data.pt
|-- policy
|   |-- __0_0.distcp
|   |-- __0_1.distcp
|   |-- common.pt
|   |-- huggingface
|   |   |-- added_tokens.json
|   |   |-- chat_template.jinja
|   |   |-- config.json
|   |   |-- generation_config.json
|   |   |-- merges.txt
|   |   |-- special_tokens_map.json
|   |   |-- tokenizer.json
|   |   |-- tokenizer_config.json
|   |   `-- vocab.json
|   `-- metadata.json
`-- trainer_state.pt

For Megatron model shards, the output directory will be created with the following structure:
.
├── added_tokens.json
├── chat_template.jinja (optional: this file is for chat specific tasks)
├── config.json
├── generation_config.json (optional: default decoding parameters)
├── merges.txt
├── model.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json

Example usage:
uv run --isolated --frozen --extra vllm scripts/convert_megatron_to_hf.py --ckpt-dir /home/aaron/ckpts/gsm8k_0.5B_ckpt/global_step_10 --out-dir /home/aaron/hf/glob_step_10
"""

import argparse
import re
import json
import sys
from pathlib import Path
import os
import shutil
from typing import List, Dict, Optional
from safetensors.torch import save_file

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
import torch

def load_metadata(policy_dir:str) -> Dict:
    """
    Load the `metadata.json` that describes the shard service.
    """
    meta_path = os.path.join(policy_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {policy_dir}")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    print(f"[INFO] Loaded metadata with keys: {list(metadata.keys())[:5]}")
    return metadata


def find_shard_files(checkpoint_dir: str) -> List[str]:
    """Find all the shard files `.distcp` and `common.pt`"""
    shard_paths = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith(".distcp") or fname == "common.pt":
            shard_paths.append(os.path.join(checkpoint_dir, fname))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {checkpoint_dir}")
    print(f"[INFO] Found {len(shard_paths)} shard files.")
    return shard_paths
    
def load_shard(path: str) -> Dict[str, torch.Tensor]:
    """Load single shard"""
    print(f"[LOAD] {path}")
    shard = torch.load(path, map_location="cpu")
    if isinstance(shard, dict):
        return shard
    else:
        raise TypeError(f"Unexpected shard type in {path}: {type(shard)}")

def merge_shards(shard_paths: List[str]) -> Dict[str, torch.Tensor]:
    """
    Merge multiple Megatron shards in a single flat `state_dict`.
    Concatenates tensors with identical keys when shapes align.
    """
    merged: Dict[str, torch.Tensor] = {}
    for path in shard_paths:
        shard = load_shard(path)
        for k, v in shard.items():
            if k not in merged:
                merged[k] = v
            else:
                try:
                    merged[k] = torch.cat([merged[k], v], dim = 0)
                except Exception as e:
                    print(f"[WARN] Could not merge the merged key {k} and value {v}. Keeping first shard only.")
    print(f"[INFO] Merged {len(merged)} unique tensors. ")
    return merged

def save_as_safetensors(state_dict: Dict[str, torch.Tensor], output_dir:str):
    """Save the merged state dict to a safetensors file"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    print(f"[SAVE] Writing {out_path} ...")
    save_file(state_dict, out_path)
    print(f"[DONE] Model saved as safetensors.")

def save_as_pt(state_dict: Dict[str, torch.Tensor], output_dir:str):
    """Fallback to pytorch .bin format"""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pytorch_model.bin")
    print(f"[SAVE] Writing {out_path} ...")
    torch.save(state_dict, out_path)
    print(f"[DONE] Model saved as pytorch_model.bin.")


## ================================================================= ##
# Main Conversion Routine #
## ================================================================= ##

def convert_megatron_to_hf(
    checkpoint_dir:str,
    output_dir:str,
    save_format:str = "safetensors", 
    verify_metadata: bool = True
):
    """
    High-level conversion pipeline that converts megatron models to hf
        1. Loads metadata (optional)
        2. Finds + Merge shards
        3. Save as Huggingface compatible weights
    """
    print(f"\n[START] Converting Megatron checkpoints -> HuggingFace format")
    print(f"  ├─ checkpoint_dir = {checkpoint_dir}")
    print(f"  ├─ output_dir     = {output_dir}")
    print(f"  └─ save_format    = {save_format}\n")

    policy_dir = checkpoint_dir + "/policy"
    if verify_metadata:
        metadata = load_metadata(policy_dir)
        print(f"Metadata is: {metadata}")
    
    shard_paths = find_shard_files(policy_dir)
    state_dict = merge_shards(shard_paths)

    if save_format == "safetensors":
        save_as_safetensors(state_dict, output_dir)
    else:
        save_as_pt(state_dict, output_dir)

    print(f"\n[VERIFY] To load: `from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('{output_dir}')`\n")

    

def main():
    ap = argparse.ArgumentParser(
        description="Convert Megatron checkpoint shards into a HuggingFace .safetensors file."
    )
    ap.add_argument(
        "--ckpt-dir", type=str, required=True, 
        help="Path to the checkpoint directory, containing trainer_state.pt (contains common.pt)"
    )
    ap.add_argument("--out-dir",  type=str, required=True, help="Output for HF model folder.")
    ap.add_argument("--save-format", type=str, default="safetensors", choices=["safetensors", "pt"], help="Output format: safetensors (recommended) or pt.")
    ap.add_argument(
        "--skip-metadata", action="store_true",
        help="Skip metadata.json validation."
    )
    ap.add_argument("--validate-load", action="store_true", help="Try loading with the Transformers Module after saving")

    args = ap.parse_args()
    
    convert_megatron_to_hf(
        checkpoint_dir=args.ckpt_dir,
        output_dir=args.out_dir,
        save_format=args.save_format,
        verify_metadata=not args.skip_metadata,
    )

if __name__ == "__main__":
    main()
