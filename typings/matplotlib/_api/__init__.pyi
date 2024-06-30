import functools
from typing import Callable, Iterable, Type
from .deprecation import (
    MatplotlibDeprecationWarning as MatplotlibDeprecationWarning,
    delete_parameter as delete_parameter,
    deprecate_method_override as deprecate_method_override,
    deprecate_privatize_attribute as deprecate_privatize_attribute,
    deprecated as deprecated,
    make_keyword_only as make_keyword_only,
    rename_parameter as rename_parameter,
    suppress_matplotlib_deprecation_warning as suppress_matplotlib_deprecation_warning,
    warn_deprecated as warn_deprecated,
)
from _typeshed import Incomplete
from collections.abc import Generator

class classproperty:
    fset: Incomplete
    fdel: Incomplete
    def __init__(
        self,
        fget: Callable[..., Incomplete],
        fset: Incomplete | None = None,
        fdel: Incomplete | None = None,
        doc: Incomplete | None = None,
    ) -> None: ...
    def __get__(self, instance: Incomplete, owner: Incomplete) -> Incomplete: ...
    @property
    def fget(self) -> Incomplete: ...

def check_isinstance(types: Incomplete, /, **kwargs: Incomplete) -> None: ...
def check_in_list(
    values: Iterable[Incomplete],
    /,
    *,
    _print_supported_values: bool = True,
    **kwargs: Incomplete,
) -> None: ...
def check_shape(shape, /, **kwargs) -> None: ...
def check_getitem(mapping: dict[Incomplete, Incomplete], /, **kwargs: Incomplete): ...
def caching_module_getattr(cls) -> functools._lru_cache_wrapper: ...
def define_aliases(
    alias_d: dict[str, list[str]], cls: Incomplete | None = None
) -> functools.partial: ...
def select_matching_signature(
    funcs: list[Callable[..., Incomplete]], *args: Incomplete, **kwargs: Incomplete
): ...
def nargs_error(name, takes, given): ...
def kwarg_error(name, kw): ...
def recursive_subclasses(cls) -> Generator[Incomplete, Incomplete, None]: ...
def warn_external(
    message: MatplotlibDeprecationWarning | PendingDeprecationWarning | str,
    category: None | Type[MatplotlibDeprecationWarning] = None,
) -> None: ...
