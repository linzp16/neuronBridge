"""Small, deterministic equation parser for maintainer-owned model specs.

The parser accepts controlled arithmetic and registered mathematical functions.
It does not evaluate Python, execute strings, or expose an extension hook.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .math_functions import MathFunctionError, math_function_symbol, validate_math_function


class EquationSyntaxError(ValueError):
    """An equation block is outside the maintained subset."""


@dataclass(frozen=True)
class Number:
    value: str


@dataclass(frozen=True)
class Identifier:
    name: str


@dataclass(frozen=True)
class Unary:
    operator: str
    operand: "Expression"


@dataclass(frozen=True)
class Binary:
    operator: str
    left: "Expression"
    right: "Expression"


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: tuple["Expression", ...]


Expression = Number | Identifier | Unary | Binary | FunctionCall


@dataclass(frozen=True)
class Comparison:
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True)
class EventStatement:
    target: str
    operator: str
    expression: Expression


@dataclass(frozen=True)
class Ode:
    state: str
    expression: Expression
    unit: str


_TOKEN = re.compile(
    r"\s*(?:(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|"
    r"([A-Za-z_][A-Za-z0-9_]*)|(.))"
)
_ODE = re.compile(
    r"^d([A-Za-z_][A-Za-z0-9_]*)/dt\s*=\s*(.+?)\s*:\s*([^\s]+)(?:\s+.*)?$")


class _ExpressionParser:
    def __init__(self, source: str) -> None:
        self._tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(source):
            match = _TOKEN.match(source, position)
            if match is None:
                raise EquationSyntaxError(f"invalid token near {source[position:]!r}")
            position = match.end()
            number, identifier, symbol = match.groups()
            if number is not None:
                self._tokens.append(("number", number))
            elif identifier is not None:
                self._tokens.append(("identifier", identifier))
            elif symbol in "+-*/(),":
                self._tokens.append(("symbol", symbol))
            else:
                raise EquationSyntaxError(f"unsupported token {symbol!r}")
        self._index = 0

    def parse(self) -> Expression:
        expression = self._parse_sum()
        if self._index != len(self._tokens):
            raise EquationSyntaxError(f"unexpected token {self._tokens[self._index][1]!r}")
        return expression

    def _parse_sum(self) -> Expression:
        expression = self._parse_product()
        while self._peek("+") or self._peek("-"):
            operator = self._consume()[1]
            expression = Binary(operator, expression, self._parse_product())
        return expression

    def _parse_product(self) -> Expression:
        expression = self._parse_unary()
        while self._peek("*") or self._peek("/"):
            operator = self._consume()[1]
            expression = Binary(operator, expression, self._parse_unary())
        return expression

    def _parse_unary(self) -> Expression:
        if self._peek("+") or self._peek("-"):
            return Unary(self._consume()[1], self._parse_unary())
        if self._peek("("):
            self._consume()
            expression = self._parse_sum()
            if not self._peek(")"):
                raise EquationSyntaxError("missing closing parenthesis")
            self._consume()
            return expression
        if self._index == len(self._tokens):
            raise EquationSyntaxError("unexpected end of expression")
        token_type, value = self._consume()
        if token_type == "number":
            return Number(value)
        if token_type == "identifier":
            if not self._peek("("):
                return Identifier(value)
            self._consume()
            arguments = []
            if not self._peek(")"):
                while True:
                    arguments.append(self._parse_sum())
                    if not self._peek(","):
                        break
                    self._consume()
            if not self._peek(")"):
                raise EquationSyntaxError(f"missing closing parenthesis for function {value!r}")
            self._consume()
            try:
                validate_math_function(value, len(arguments))
            except MathFunctionError as error:
                raise EquationSyntaxError(str(error)) from error
            return FunctionCall(value, tuple(arguments))
        raise EquationSyntaxError(f"expected value, got {value!r}")

    def _peek(self, symbol: str) -> bool:
        return self._index < len(self._tokens) and self._tokens[self._index] == ("symbol", symbol)

    def _consume(self) -> tuple[str, str]:
        token = self._tokens[self._index]
        self._index += 1
        return token


def parse_expression(source: str) -> Expression:
    return _ExpressionParser(source).parse()


def parse_equations(source: str) -> tuple[Ode, ...]:
    equations: list[Ode] = []
    states: set[str] = set()
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _ODE.match(line)
        if match is None:
            raise EquationSyntaxError(f"unsupported equation declaration {line!r}")
        state, expression, unit = match.groups()
        if state in states:
            raise EquationSyntaxError(f"duplicate differential state {state!r}")
        states.add(state)
        equations.append(Ode(state, _ExpressionParser(expression).parse(), unit))
    if not equations:
        raise EquationSyntaxError("equation block contains no differential equation")
    return tuple(equations)


def emit_cpp(
    expression: Expression,
    identifiers: dict[str, str],
    target: str = "cpu",
) -> str:
    if isinstance(expression, Number):
        return expression.value + (
            "f" if any(marker in expression.value for marker in ".eE") else ".0f"
        )
    if isinstance(expression, Identifier):
        try:
            return identifiers[expression.name]
        except KeyError as error:
            raise EquationSyntaxError(f"unknown identifier {expression.name!r}") from error
    if isinstance(expression, Unary):
        return f"({expression.operator}{emit_cpp(expression.operand, identifiers, target)})"
    if isinstance(expression, FunctionCall):
        try:
            symbol = math_function_symbol(expression.name, len(expression.arguments), target)
        except MathFunctionError as error:
            raise EquationSyntaxError(str(error)) from error
        arguments = ", ".join(
            emit_cpp(argument, identifiers, target) for argument in expression.arguments
        )
        return f"{symbol}({arguments})"
    return (
        f"({emit_cpp(expression.left, identifiers, target)} {expression.operator} "
        f"{emit_cpp(expression.right, identifiers, target)})"
    )


def identifiers_in(expression: Expression) -> frozenset[str]:
    if isinstance(expression, Number):
        return frozenset()
    if isinstance(expression, Identifier):
        return frozenset((expression.name,))
    if isinstance(expression, Unary):
        return identifiers_in(expression.operand)
    if isinstance(expression, FunctionCall):
        identifiers = frozenset()
        for argument in expression.arguments:
            identifiers |= identifiers_in(argument)
        return identifiers
    return identifiers_in(expression.left) | identifiers_in(expression.right)


def parse_comparison(source: str) -> Comparison:
    depth = 0
    found: tuple[int, str] | None = None
    index = 0
    while index < len(source):
        character = source[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise EquationSyntaxError("unbalanced comparison parentheses")
        elif depth == 0:
            operator = next(
                (candidate for candidate in (">=", "<=", "==", "!=", ">", "<")
                 if source.startswith(candidate, index)),
                None,
            )
            if operator is not None:
                if found is not None:
                    raise EquationSyntaxError("comparison chaining is not supported")
                found = (index, operator)
                index += len(operator) - 1
        index += 1
    if depth != 0:
        raise EquationSyntaxError("unbalanced comparison parentheses")
    if found is None:
        raise EquationSyntaxError("event condition requires one comparison operator")
    position, operator = found
    left = source[:position].strip()
    right = source[position + len(operator):].strip()
    if not left or not right:
        raise EquationSyntaxError("comparison operands must not be empty")
    return Comparison(operator, parse_expression(left), parse_expression(right))


def emit_comparison_cpp(
    comparison: Comparison,
    identifiers: dict[str, str],
    target: str = "cpu",
) -> str:
    return (
        f"{emit_cpp(comparison.left, identifiers, target)} {comparison.operator} "
        f"{emit_cpp(comparison.right, identifiers, target)}"
    )


def parse_event_statement(source: str) -> EventStatement:
    match = re.fullmatch(
        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\+=|-=|=)\s*(.+?)\s*", source
    )
    if match is None:
        raise EquationSyntaxError(f"invalid event statement {source!r}")
    return EventStatement(match[1], match[2], parse_expression(match[3]))
