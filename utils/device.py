# utils/device.py
import torch

def pick_device():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    return device