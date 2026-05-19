import torch


type Sample = tuple[torch.Tensor, int, str, int, int]
type Input = tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]
type Outputs = torch.Tensor
