"""Maintainer-owned mathematical function registry for equation lowering."""

from __future__ import annotations

from dataclasses import dataclass


class MathFunctionError(ValueError):
    """A function call is outside the maintained equation subset."""


@dataclass(frozen=True)
class MathFunction:
    min_arity: int
    max_arity: int
    cpu_symbol: str
    cuda_symbol: str


MATH_FUNCTIONS = {
    "exp": MathFunction(1, 1, "std::exp", "expf"),
    "expm1": MathFunction(1, 1, "std::expm1", "expm1f"),
    "log": MathFunction(1, 1, "std::log", "logf"),
    "log1p": MathFunction(1, 1, "std::log1p", "log1pf"),
    "sqrt": MathFunction(1, 1, "std::sqrt", "sqrtf"),
    "abs": MathFunction(1, 1, "std::fabs", "fabsf"),
    "min": MathFunction(2, 2, "std::fmin", "fminf"),
    "max": MathFunction(2, 2, "std::fmax", "fmaxf"),
    "pow": MathFunction(2, 2, "std::pow", "powf"),
}


def validate_math_function(name: str, argument_count: int) -> MathFunction:
    function = MATH_FUNCTIONS.get(name)
    if function is None:
        raise MathFunctionError(f"unsupported mathematical function {name!r}")
    if not function.min_arity <= argument_count <= function.max_arity:
        expected = (
            str(function.min_arity)
            if function.min_arity == function.max_arity
            else f"{function.min_arity}..{function.max_arity}"
        )
        raise MathFunctionError(
            f"function {name!r} expects {expected} arguments, got {argument_count}"
        )
    return function


def math_function_symbol(name: str, argument_count: int, target: str) -> str:
    function = validate_math_function(name, argument_count)
    if target == "cpu":
        return function.cpu_symbol
    if target == "cuda":
        return function.cuda_symbol
    raise MathFunctionError(f"unsupported expression target {target!r}")
