import ast
import operator
from typing import Any, Union
from PRIMORA.prime.contract import PrimeResult
from PRIMORA.specialists.specialist import Specialist


Number = Union[int, float]

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _evaluate(node: ast.AST) -> Number:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported literal: {type(node.value).__name__}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return +_evaluate(node.operand)
        if isinstance(node.op, ast.USub):
            return -_evaluate(node.operand)
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
    raise ValueError(f"Unsupported expression: {type(node).__name__}")


class CalculateSpecialist(Specialist):
    """Deterministic arithmetic evaluation. No model calls, no eval()."""

    name = "calculate"

    def handle(self, payload: dict[str, Any]) -> PrimeResult:
        expression = payload.get("expression") if isinstance(payload, dict) else None
        if not isinstance(expression, str) or not expression.strip():
            return PrimeResult.failure("Missing or invalid 'expression' in payload")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return PrimeResult.failure(f"Invalid expression: {e}")
        try:
            result = _evaluate(tree.body)
        except ZeroDivisionError:
            return PrimeResult.failure("Division by zero")
        except ValueError as e:
            return PrimeResult.failure(str(e))
        except (OverflowError, ArithmeticError) as e:
            return PrimeResult.failure(f"Arithmetic error: {e}")
        return PrimeResult.success({"result": result})
