import torch

def get_device_type():
    device_info = torch.cuda.get_device_name(0)
    supported_device = ['A800', 'A100', 'H20', '3090', '4090']
    for device in supported_device:
        if device in device_info:
            return device
    return None