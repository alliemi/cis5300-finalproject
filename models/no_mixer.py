"""Control model with no attention: each block only has the shared 4-token
convolution and the MLP. It shows how far the task can be solved without
any long-range memory, and gives the harness tests a model-independent
mixer."""

import torch
import torch.nn as nn


class NoMixer(nn.Module):
    requires_cuda = False

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
