import yaml
from pprint import pprint
import json
import logging
import torch
import os

def load_yaml_file(config_file, show = True): 

    with open(config_file, "r") as f:
       print(f"* Loading YAML file: {config_file}")
       config = yaml.safe_load(f)

    if show:
        pprint(config)

    return config

def load_json_file(json_file):

    with open(json_file, 'r') as file:
        data = json.load(file)

    return data

def save_checkpoint(checkpoint_file, model, vocab, EMBED_DIM, OUTPUT_DIM):
    # Now save the artifacts of the training
    print(f"Saving checkpoint to {checkpoint_file}...")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok = True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "vocab": vocab,
        "embed_dim": EMBED_DIM,
        "output_dim": OUTPUT_DIM,
    }
    torch.save(checkpoint, checkpoint_file)