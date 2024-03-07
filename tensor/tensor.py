import numpy as np
from numpy import ndarray as array
from backward import *
from typing import List,Callable

class Tensor:
    _data:np.ndarray
    _requires_grad:bool=False
    _grad:array=None
    _variables:List['Tensor']=None
    _grad_fn:Callable[[*'Tensor'],array]|None=None

    @staticmethod
    def random(*shape)->'Tensor':
        data =  np.random.randn(*shape)
        return Tensor(data)

    @staticmethod
    def matmul(matrix_a:'Tensor',matrix_b:'Tensor')->'Tensor':
        result = np.matmul(matrix_a._data,matrix_b._data)
        tensor = Tensor(result)
    
    @staticmethod
    def ones(*shape:int)->'Tensor':
        tensor=Tensor(np.ones(shape,dtype=np.float32))
        return tensor
    def zeros(*shape:int)->'Tensor':
        return Tensor(np.zeros(shape,dtype=np.float32))
    
    def __init__(self,data:array|List) -> None:
        if isinstance(data,List):
            data=array(data)
        self._data=data
        self._grad=None

    def _requires_grad(self,requires:bool=True)->None:
        self._requires_grad=requires
    
    def shape(self)->array.shape:
        return self._data.shape
    
    def backward(self)->array:
        if self._requires_grad and self._grad_fn and self._variables is not None:
            self._grad= self._grad_fn(self._variables)
            return self._grad
        else:
            return 0

    def step(self,lr:float)->None:
        if self._grad:
            assert self._grad.shape==self._data.shape
            self._data -= self._grad*lr

    def __add__(self,other:'Tensor')->'Tensor':
        assert self.shape()==other.shape()
        result = self._data+other._data
        t = Tensor(result)
        if self._requires_grad or other._requires_grad:
            t._requires_grad()
            t._grad_fn=backward__add
            result.variables=[self,other]
        return result

    def __sub__(self,other:'Tensor')->'Tensor':
        assert self.shape()==other.shape()
        result = Tensor(self._data-other._data)
        if self._requires_grad or other._requires_grad:
            result._requires_grad()
            result._grad_fn=backward__sub
            result._variables=[self,other]
        return result

    def __mul__(self,other:'Tensor')->'Tensor':
        result = Tensor(np.matmul(self._data,other._data))
        if self._requires_grad or other._requires_grad:
            result._requires_grad=True
            result._grad_fn=backward__matmul
            result._variables=[self,other]
        return result

if __name__=="__main__":
    a=Tensor.ones(12,3)
    b=Tensor.random(3,6)
    b._requires_grad=True
    target = Tensor.random(12,6)
    
    epoch=12
    while(epoch):
        out = a*b
        #loss

        #MSE
        grad = (target-out)
        out.backward(grad)



