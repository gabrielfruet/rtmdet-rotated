from jaxtyping import jaxtyped
from beartype import beartype as beartype_typechecker
from functools import wraps


def typechecker(fn):
    return jaxtyped(typechecker=beartype_typechecker)(fn)
