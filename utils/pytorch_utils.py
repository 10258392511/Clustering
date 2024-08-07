import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()
