"""
This type stub file was generated by pyright.
"""

_TORCHFUNCTION_SUBCLASS = ...
class _ReturnTypeCM:
    def __init__(self, to_restore) -> None:
        ...
    
    def __enter__(self): # -> Self:
        ...
    
    def __exit__(self, *args): # -> None:
        ...
    


def set_return_type(return_type: str): # -> _ReturnTypeCM:
    """Set the return type of torch operations on :class:`~torchvision.tv_tensors.TVTensor`.

    This only affects the behaviour of torch operations. It has no effect on
    ``torchvision`` transforms or functionals, which will always return as
    output the same type that was passed as input.

    .. warning::

        We recommend using :class:`~torchvision.transforms.v2.ToPureTensor` at
        the end of your transform pipelines if you use
        ``set_return_type("TVTensor")``. This will avoid the
        ``__torch_function__`` overhead in the models ``forward()``.

    Can be used as a global flag for the entire program:

    .. code:: python

        img = tv_tensors.Image(torch.rand(3, 5, 5))
        img + 2  # This is a pure Tensor (default behaviour)

        set_return_type("TVTensor")
        img + 2  # This is an Image

    or as a context manager to restrict the scope:

    .. code:: python

        img = tv_tensors.Image(torch.rand(3, 5, 5))
        img + 2  # This is a pure Tensor
        with set_return_type("TVTensor"):
            img + 2  # This is an Image
        img + 2  # This is a pure Tensor

    Args:
        return_type (str): Can be "TVTensor" or "Tensor" (case-insensitive).
            Default is "Tensor" (i.e. pure :class:`torch.Tensor`).
    """
    ...

_FORCE_TORCHFUNCTION_SUBCLASS = ...
