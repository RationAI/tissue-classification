import torch


type Sample = tuple[torch.Tensor, int, str]
type Input = tuple[torch.Tensor, torch.Tensor, list[str]]
type Outputs = torch.Tensor
