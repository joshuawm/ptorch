from typing import Any
from layer import Module
from tensor import Tensor

class Linear(Module):
    def __init__(self,inchannel:int,outchannel:int) -> None:
        super().__init__()
        self.weight=Tensor.random(inchannel,outchannel)
        self.weight.requires_grad()

    def forward(self, input: Tensor) ->Tensor:
        return self*input
    

