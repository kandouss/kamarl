import torch


def find_cuda_device(device_name):
    cuda_devices = {torch.cuda.get_device_name(f'cuda:{x}'):x for x in range(torch.cuda.device_count())}
    matching_cuda_devices = [x for x in cuda_devices.keys() if device_name in x]

    if len(matching_cuda_devices) == 0:
        return torch.device('cpu')
    return torch.device(f'cuda:{cuda_devices[matching_cuda_devices[0]]}')
