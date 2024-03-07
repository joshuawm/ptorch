from typing import Any


class Module:
    def __init__(self) -> None:
        pass
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.forward(args)
    def forward(self,*args:Any)->Any:
        pass
    