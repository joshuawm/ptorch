from tensor import Tensor
from numpy import ndarray as array
import numpy as np


def backward__add(*tensors:Tensor)->array:
    grad=None
    for tensor in tensors:
        g =  tensor.backward()
        if g and grad:
            grad+=g
        if g and grad is None:
            grad=g
    return grad

def backward__sub(*tensors:Tensor)->array:
    grad=None
    for tensor in tensors:
        g = tensor.backward()
        if g and grad:
            grad -= g
        if g and grad is None:
            grad = g
    return grad

def backward__matmul(tensor_a:Tensor,tensor_b:Tensor)->array:
    grad = np.matmul(tensor_a._data.transpose(),tensor_b._data)
    return grad

