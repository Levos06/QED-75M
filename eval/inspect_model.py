import torch
from pathlib import Path

path = "model.pt"
checkpoint = torch.load(path, map_location="cpu")
print("Keys in checkpoint:", checkpoint.keys())
print("Model Config:", checkpoint.get("model_config"))
if "train_config" in checkpoint:
    print("Train Config:", checkpoint["train_config"])
