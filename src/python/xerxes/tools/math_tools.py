# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Math tools module for Xerxes.

Exports:
    - Calculator
    - StatisticalAnalyzer
    - MathematicalFunctions
    - NumberTheory
    - UnitConverter"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from collections.abc import Callable
from decimal import Decimal, getcontext
from typing import Any

from ..types import AgentBaseFn

_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
    "pow": pow,
    "exp": math.exp,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
_ALLOWED_BINOPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type, Callable[..., Any]] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expression: str) -> float:
    """Internal helper to safe eval.

    Args:
        expression (str): IN: expression. OUT: Consumed during execution.
    Returns:
        float: OUT: Result of the operation."""

    tree = ast.parse(expression, mode="eval")

    def _eval(node: ast.AST) -> float:
        """Internal helper to eval.

        Args:
            node (ast.AST): IN: node. OUT: Consumed during execution.
        Returns:
            float: OUT: Result of the operation."""
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Only numeric constants are allowed (got {type(node.value).__name__})")
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_NAMES:
                return _ALLOWED_NAMES[node.id]
            raise ValueError(f"Name {node.id!r} is not allowed")
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _ALLOWED_BINOPS:
                raise ValueError(f"Operator {type(node.op).__name__} is not allowed")
            return _ALLOWED_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _ALLOWED_UNARYOPS:
                raise ValueError(f"Unary operator {type(node.op).__name__} is not allowed")
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            func_name = node.func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError(f"Function {func_name!r} is not allowed")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed")
            return _ALLOWED_FUNCS[func_name](*[_eval(a) for a in node.args])
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

    return _eval(tree)


class Calculator(AgentBaseFn):
    """Calculator.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        expression: str | None = None,
        operation: str | None = None,
        operands: list[float] | None = None,
        precision: int = 10,
        **context_variables,
    ) -> dict[str, Any]:
        """Static call.

        Args:
            expression (str | None, optional): IN: expression. Defaults to None. OUT: Consumed during execution.
            operation (str | None, optional): IN: operation. Defaults to None. OUT: Consumed during execution.
            operands (list[float] | None, optional): IN: operands. Defaults to None. OUT: Consumed during execution.
            precision (int, optional): IN: precision. Defaults to 10. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        result: dict[str, Any] = {}
        getcontext().prec = precision

        if expression:
            try:
                value = _safe_eval(expression)

                result["expression"] = expression
                result["result"] = float(value)
                result["decimal_result"] = str(Decimal(str(value)))

            except Exception as e:
                return {"error": f"Invalid expression: {e!s}"}

        elif operation and operands:
            try:
                if operation == "add":
                    value = sum(operands)
                elif operation == "multiply":
                    value = 1
                    for x in operands:
                        value *= x
                elif operation == "mean":
                    value = statistics.mean(operands)
                elif operation == "median":
                    value = statistics.median(operands)
                elif operation == "mode":
                    try:
                        value = statistics.mode(operands)
                    except statistics.StatisticsError:
                        value = None
                        result["note"] = "No unique mode found"
                elif operation == "stdev":
                    value = statistics.stdev(operands) if len(operands) > 1 else 0
                elif operation == "variance":
                    value = statistics.variance(operands) if len(operands) > 1 else 0
                elif operation == "min":
                    value = min(operands)
                elif operation == "max":
                    value = max(operands)
                elif operation == "range":
                    value = max(operands) - min(operands)
                elif operation == "sum_of_squares":
                    value = sum(x**2 for x in operands)
                elif operation == "root_mean_square":
                    value = math.sqrt(sum(x**2 for x in operands) / len(operands))
                elif operation == "geometric_mean":
                    product: float = 1
                    for x in operands:
                        if x <= 0:
                            return {"error": "Geometric mean requires positive numbers"}
                        product *= x
                    value = product ** (1 / len(operands))
                elif operation == "harmonic_mean":
                    value = statistics.harmonic_mean(operands)
                else:
                    return {"error": f"Unknown operation: {operation}"}

                result["operation"] = operation
                result["operands"] = operands
                result["result"] = value
                result["count"] = len(operands)

            except Exception as e:
                return {"error": f"Calculation failed: {e!s}"}

        else:
            return {"error": "Either expression or operation with operands required"}

        return result


class StatisticalAnalyzer(AgentBaseFn):
    """Statistical analyzer.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        data: list[float],
        analysis_type: str = "descriptive",
        confidence_level: float = 0.95,
        **context_variables,
    ) -> dict[str, Any]:
        """Static call.

        Args:
            data (list[float]): IN: data. OUT: Consumed during execution.
            analysis_type (str, optional): IN: analysis type. Defaults to 'descriptive'. OUT: Consumed during execution.
            confidence_level (float, optional): IN: confidence level. Defaults to 0.95. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        if not data:
            return {"error": "Data cannot be empty"}

        result: dict[str, Any] = {"data_points": len(data)}

        if analysis_type == "descriptive":
            result["statistics"] = {
                "count": len(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data),
                "sum": sum(data),
            }

            if len(data) > 1:
                result["statistics"]["std_dev"] = statistics.stdev(data)
                result["statistics"]["variance"] = statistics.variance(data)

            try:
                result["statistics"]["mode"] = statistics.mode(data)
            except statistics.StatisticsError:
                result["statistics"]["mode"] = None

            quantile_list = statistics.quantiles(data, n=4)
            result["quartiles"] = {
                "Q1": quantile_list[0],
                "Q2": quantile_list[1],
                "Q3": quantile_list[2],
            }

            result["quartiles"]["IQR"] = result["quartiles"]["Q3"] - result["quartiles"]["Q1"]

            iqr = result["quartiles"]["IQR"]
            lower_bound = result["quartiles"]["Q1"] - 1.5 * iqr
            upper_bound = result["quartiles"]["Q3"] + 1.5 * iqr

            outliers = [x for x in data if x < lower_bound or x > upper_bound]
            result["outliers"] = {
                "count": len(outliers),
                "values": outliers[:20],
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

        elif analysis_type == "distribution":
            sorted(data)
            mean = statistics.mean(data)

            if len(data) > 2:
                std_dev = statistics.stdev(data)
                n = len(data)
                skewness = sum((x - mean) ** 3 for x in data) / (n * std_dev**3)
                result["skewness"] = skewness

                kurtosis = sum((x - mean) ** 4 for x in data) / (n * std_dev**4) - 3
                result["kurtosis"] = kurtosis

            num_bins = min(10, len(set(data)))
            if num_bins > 1:
                data_range = max(data) - min(data)
                bin_width = data_range / num_bins

                bins = []
                for i in range(num_bins):
                    bin_start = min(data) + i * bin_width
                    bin_end = bin_start + bin_width
                    count = sum(1 for x in data if bin_start <= x < bin_end or (i == num_bins - 1 and x == bin_end))
                    bins.append(
                        {
                            "range": f"{bin_start:.2f} - {bin_end:.2f}",
                            "count": count,
                            "frequency": count / len(data),
                        }
                    )

                result["frequency_distribution"] = bins

        elif analysis_type == "correlation":
            if len(data) % 2 != 0:
                return {"error": "Correlation analysis requires paired data (even number of values)"}

            n = len(data) // 2
            x_data = data[:n]
            y_data = data[n:]

            mean_x = statistics.mean(x_data)
            mean_y = statistics.mean(y_data)

            numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_data, y_data, strict=False))
            sum_sq_x = sum((x - mean_x) ** 2 for x in x_data)
            sum_sq_y = sum((y - mean_y) ** 2 for y in y_data)

            if sum_sq_x * sum_sq_y > 0:
                correlation = numerator / math.sqrt(sum_sq_x * sum_sq_y)
                result["correlation"] = {
                    "pearson_r": correlation,
                    "r_squared": correlation**2,
                    "strength": "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak",
                    "direction": "positive" if correlation > 0 else "negative" if correlation < 0 else "none",
                }
            else:
                result["correlation"] = {"error": "Cannot calculate correlation (zero variance)"}

        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

        return result


class MathematicalFunctions(AgentBaseFn):
    """Mathematical functions.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        function: str,
        input_value: float | None = None,
        parameters: dict[str, float] | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Static call.

        Args:
            function (str): IN: function. OUT: Consumed during execution.
            input_value (float | None, optional): IN: input value. Defaults to None. OUT: Consumed during execution.
            parameters (dict[str, float] | None, optional): IN: parameters. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        result: dict[str, Any] = {}

        if input_value is None:
            return {"error": "input_value required"}

        try:
            if function == "sin":
                value = math.sin(input_value)
            elif function == "cos":
                value = math.cos(input_value)
            elif function == "tan":
                value = math.tan(input_value)
            elif function == "asin":
                if -1 <= input_value <= 1:
                    value = math.asin(input_value)
                else:
                    return {"error": "asin input must be between -1 and 1"}
            elif function == "acos":
                if -1 <= input_value <= 1:
                    value = math.acos(input_value)
                else:
                    return {"error": "acos input must be between -1 and 1"}
            elif function == "atan":
                value = math.atan(input_value)
            elif function == "log":
                if input_value > 0:
                    base = parameters.get("base", math.e) if parameters else math.e
                    if base == math.e:
                        value = math.log(input_value)
                    else:
                        value = math.log(input_value, base)
                else:
                    return {"error": "log input must be positive"}
            elif function == "log10":
                if input_value > 0:
                    value = math.log10(input_value)
                else:
                    return {"error": "log10 input must be positive"}
            elif function == "exp":
                value = math.exp(input_value)
            elif function == "sqrt":
                if input_value >= 0:
                    value = math.sqrt(input_value)
                else:
                    return {"error": "sqrt input must be non-negative"}
            elif function == "abs":
                value = abs(input_value)
            elif function == "floor":
                value = math.floor(input_value)
            elif function == "ceil":
                value = math.ceil(input_value)
            elif function == "round":
                decimals = int(parameters.get("decimals", 0)) if parameters else 0
                value = round(input_value, decimals)
            elif function == "factorial":
                if input_value == int(input_value) and input_value >= 0:
                    value = math.factorial(int(input_value))
                else:
                    return {"error": "factorial input must be non-negative integer"}
            elif function == "pow":
                exponent = parameters.get("exponent", 2) if parameters else 2
                value = math.pow(input_value, exponent)
            elif function == "sinh":
                value = math.sinh(input_value)
            elif function == "cosh":
                value = math.cosh(input_value)
            elif function == "tanh":
                value = math.tanh(input_value)
            else:
                return {"error": f"Unknown function: {function}"}

            result["function"] = function
            result["input"] = input_value
            result["result"] = value

            if parameters:
                result["parameters"] = parameters

        except Exception as e:
            return {"error": f"Function evaluation failed: {e!s}"}

        return result


class NumberTheory(AgentBaseFn):
    """Number theory.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        operation: str,
        number: int | None = None,
        numbers: list[int] | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Static call.

        Args:
            operation (str): IN: operation. OUT: Consumed during execution.
            number (int | None, optional): IN: number. Defaults to None. OUT: Consumed during execution.
            numbers (list[int] | None, optional): IN: numbers. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        result: dict[str, Any] = {}

        if operation == "prime":
            if number is None:
                return {"error": "number required for prime check"}

            def is_prime(n):
                """Check whether prime.

                Args:
                    n (Any): IN: n. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                if n < 2:
                    return False
                if n == 2:
                    return True
                if n % 2 == 0:
                    return False
                for i in range(3, int(math.sqrt(n)) + 1, 2):
                    if n % i == 0:
                        return False
                return True

            result["number"] = number
            result["is_prime"] = is_prime(number)

            if is_prime(number):
                result["type"] = "prime"
            else:
                result["type"] = "composite" if number > 1 else "neither"

        elif operation == "factors":
            if number is None:
                return {"error": "number required for factorization"}

            def get_factors(n):
                """Retrieve the factors.

                Args:
                    n (Any): IN: n. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                factors = []
                for i in range(1, int(math.sqrt(abs(n))) + 1):
                    if n % i == 0:
                        factors.append(i)
                        if i != n // i:
                            factors.append(n // i)
                return sorted(factors)

            def prime_factors(n):
                """Prime factors.

                Args:
                    n (Any): IN: n. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                factors = []
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        factors.append(d)
                        n //= d
                    d += 1
                if n > 1:
                    factors.append(n)
                return factors

            result["number"] = number
            result["factors"] = get_factors(abs(number))
            result["prime_factors"] = prime_factors(abs(number))
            result["factor_count"] = len(result["factors"])

        elif operation == "gcd":
            if not numbers or len(numbers) < 2:
                return {"error": "At least 2 numbers required for GCD"}

            def gcd(a, b):
                """Gcd.

                Args:
                    a (Any): IN: a. OUT: Consumed during execution.
                    b (Any): IN: b. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                while b:
                    a, b = b, a % b
                return a

            def gcd_multiple(nums):
                """Gcd multiple.

                Args:
                    nums (Any): IN: nums. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                result = nums[0]
                for i in range(1, len(nums)):
                    result = gcd(result, nums[i])
                return result

            result["numbers"] = numbers
            result["gcd"] = gcd_multiple(numbers)

        elif operation == "lcm":
            if not numbers or len(numbers) < 2:
                return {"error": "At least 2 numbers required for LCM"}

            def gcd(a, b):
                """Gcd.

                Args:
                    a (Any): IN: a. OUT: Consumed during execution.
                    b (Any): IN: b. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                while b:
                    a, b = b, a % b
                return a

            def lcm(a, b):
                """Lcm.

                Args:
                    a (Any): IN: a. OUT: Consumed during execution.
                    b (Any): IN: b. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                return abs(a * b) // gcd(a, b)

            def lcm_multiple(nums):
                """Lcm multiple.

                Args:
                    nums (Any): IN: nums. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                result = nums[0]
                for i in range(1, len(nums)):
                    result = lcm(result, nums[i])
                return result

            result["numbers"] = numbers
            result["lcm"] = lcm_multiple(numbers)

        elif operation == "fibonacci":
            if number is None:
                return {"error": "number required for Fibonacci sequence"}

            def fibonacci_sequence(n):
                """Fibonacci sequence.

                Args:
                    n (Any): IN: n. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                if n <= 0:
                    return []
                elif n == 1:
                    return [0]
                elif n == 2:
                    return [0, 1]

                fib = [0, 1]
                for i in range(2, n):
                    fib.append(fib[i - 1] + fib[i - 2])
                return fib

            result["length"] = number
            result["sequence"] = fibonacci_sequence(number)

            if number > 0:
                result["nth_fibonacci"] = result["sequence"][-1]

        elif operation == "collatz":
            if number is None:
                return {"error": "number required for Collatz sequence"}

            def collatz_sequence(n):
                """Collatz sequence.

                Args:
                    n (Any): IN: n. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                sequence = [n]
                while n != 1:
                    if n % 2 == 0:
                        n = n // 2
                    else:
                        n = 3 * n + 1
                    sequence.append(n)

                    if len(sequence) > 1000:
                        break
                return sequence

            result["starting_number"] = number
            result["sequence"] = collatz_sequence(number)
            result["steps"] = len(result["sequence"]) - 1
            result["max_value"] = max(result["sequence"])

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class UnitConverter(AgentBaseFn):
    """Unit converter.

    Inherits from: AgentBaseFn
    """

    @staticmethod
    def static_call(
        value: float,
        from_unit: str,
        to_unit: str,
        category: str | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Static call.

        Args:
            value (float): IN: value. OUT: Consumed during execution.
            from_unit (str): IN: from unit. OUT: Consumed during execution.
            to_unit (str): IN: to unit. OUT: Consumed during execution.
            category (str | None, optional): IN: category. Defaults to None. OUT: Consumed during execution.
            **context_variables: IN: Additional keyword arguments. OUT: Passed through to downstream calls.
        Returns:
            dict[str, Any]: OUT: Result of the operation."""

        result: dict[str, Any] = {}

        conversions = {
            "length": {
                "meter": 1.0,
                "m": 1.0,
                "centimeter": 0.01,
                "cm": 0.01,
                "millimeter": 0.001,
                "mm": 0.001,
                "kilometer": 1000.0,
                "km": 1000.0,
                "inch": 0.0254,
                "in": 0.0254,
                "foot": 0.3048,
                "ft": 0.3048,
                "yard": 0.9144,
                "yd": 0.9144,
                "mile": 1609.344,
                "mi": 1609.344,
            },
            "weight": {
                "gram": 1.0,
                "g": 1.0,
                "kilogram": 1000.0,
                "kg": 1000.0,
                "pound": 453.592,
                "lb": 453.592,
                "ounce": 28.3495,
                "oz": 28.3495,
                "stone": 6350.29,
                "ton": 1000000.0,
            },
            "volume": {
                "liter": 1.0,
                "l": 1.0,
                "milliliter": 0.001,
                "ml": 0.001,
                "gallon": 3.78541,
                "gal": 3.78541,
                "quart": 0.946353,
                "qt": 0.946353,
                "pint": 0.473176,
                "pt": 0.473176,
                "cup": 0.236588,
                "fluid_ounce": 0.0295735,
                "fl_oz": 0.0295735,
            },
            "area": {
                "square_meter": 1.0,
                "m2": 1.0,
                "square_centimeter": 0.0001,
                "cm2": 0.0001,
                "square_kilometer": 1000000.0,
                "km2": 1000000.0,
                "square_foot": 0.092903,
                "ft2": 0.092903,
                "acre": 4046.86,
                "hectare": 10000.0,
            },
            "speed": {
                "meter_per_second": 1.0,
                "mps": 1.0,
                "kilometer_per_hour": 0.277778,
                "kmh": 0.277778,
                "kph": 0.277778,
                "mile_per_hour": 0.44704,
                "mph": 0.44704,
                "knot": 0.514444,
                "kt": 0.514444,
            },
        }

        if from_unit.lower() in ["celsius", "c", "fahrenheit", "f", "kelvin", "k"]:

            def convert_temperature(val, from_u, to_u):
                """Convert temperature.

                Args:
                    val (Any): IN: val. OUT: Consumed during execution.
                    from_u (Any): IN: from u. OUT: Consumed during execution.
                    to_u (Any): IN: to u. OUT: Consumed during execution.
                Returns:
                    Any: OUT: Result of the operation."""
                from_u = from_u.lower()
                to_u = to_u.lower()

                if from_u in ["fahrenheit", "f"]:
                    celsius = (val - 32) * 5 / 9
                elif from_u in ["kelvin", "k"]:
                    celsius = val - 273.15
                else:
                    celsius = val

                if to_u in ["fahrenheit", "f"]:
                    return celsius * 9 / 5 + 32
                elif to_u in ["kelvin", "k"]:
                    return celsius + 273.15
                else:
                    return celsius

            converted = convert_temperature(value, from_unit, to_unit)
            result["value"] = value
            result["from_unit"] = from_unit
            result["to_unit"] = to_unit
            result["result"] = converted
            result["category"] = "temperature"

        else:
            if not category:
                for cat, units in conversions.items():
                    if from_unit.lower() in units and to_unit.lower() in units:
                        category = cat
                        break

            if not category:
                return {"error": f"Could not determine category for units {from_unit} and {to_unit}"}

            if category not in conversions:
                return {"error": f"Unknown category: {category}"}

            from_factor = conversions[category].get(from_unit.lower())
            to_factor = conversions[category].get(to_unit.lower())

            if from_factor is None:
                return {"error": f"Unknown unit: {from_unit} in category {category}"}
            if to_factor is None:
                return {"error": f"Unknown unit: {to_unit} in category {category}"}

            base_value = value * from_factor
            converted = base_value / to_factor

            result["value"] = value
            result["from_unit"] = from_unit
            result["to_unit"] = to_unit
            result["result"] = converted
            result["category"] = category

        return result
