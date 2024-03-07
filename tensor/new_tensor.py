import numpy as np
from numpy import ndarray as array

class Tensor:
    data:array
    _requires_grad:bool=False
    _grad_fn:callable[['Tensor'],array]=None
    _grad:array=None
    forward_nodes:list['Tensor']=[]

    def __init__(self,data:array|int|float,requires_grad:bool=False) -> None:
        self._requires_grad=requires_grad
        if isinstance(data,array):
            self.data=data
        else:
            self.data=array(data)
    #Operator: +
    def __add__(self,other:'Tensor'|float|int):
        result = self.data+other.data
        tensor = Tensor(result)
        if self._requires_grad:
            tensor.forward_nodes

    #Operator: -
    def __sub__(self,other:'Tensor'|float|int):
        result = self.data - other.data
    #Operator: *
    def __mul__(self,other:'Tensor'|float|int):
        pass 
    #Operator: @
    def __matmul__(self,other:'Tensor'|float|int):
        pass


        