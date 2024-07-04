from typing import Any

from numpy import ndarray

class Bunch(dict[Any, Any]):
    def __init__(self, **kwargs: Any) -> None: ...
    def __setattr__(
        self, key: str, value: list[str] | ndarray[Any, Any] | str
    ) -> None: ...
    def __dir__(self) -> list[str]: ...
    def __getattr__(self, key: str) -> Any: ...
    def __setstate__(self, state: Any) -> Any: ...