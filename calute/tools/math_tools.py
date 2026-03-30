# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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


"""Mathematical and statistical tools for calculations and analysis.

This module provides a comprehensive set of mathematical tools
for the Calute framework. It includes:
- Advanced calculator with expression evaluation and basic operations
- Statistical analysis with descriptive statistics and correlation
- Mathematical functions including trigonometric, logarithmic, and more
- Number theory operations like prime checking, factorization, GCD, and LCM
- Unit conversion between different measurement systems

Each tool is implemented as a class inheriting from AgentBaseFn,
making them directly usable as agent tools for mathematical computations.

Example:
    >>> calc = Calculator()
    >>> result = calc(expression="sqrt(16) + pow(2, 3)")
    >>> print(result["result"])  # 12.0
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal, getcontext
from typing import Any

from ..types import AgentBaseFn


class Calculator(AgentBaseFn):
    """Advanced calculator with various mathematical operations.

    Provides two modes of operation: expression evaluation using
    a safe subset of mathematical functions, and named operations
    on lists of operands. Supports configurable decimal precision.

    Expression mode supports:
        sin, cos, tan, log, sqrt, abs, pow, exp and basic arithmetic.

    Operation mode supports:
        add, multiply, mean, median, mode, stdev, variance, min, max,
        range, sum_of_squares, root_mean_square, geometric_mean, harmonic_mean.
    """

    @staticmethod
    def static_call(
        expression: str | None = None,
        operation: str | None = None,
        operands: list[float] | None = None,
        precision: int = 10,
        **context_variables,
    ) -> dict[str, Any]:
        """Perform mathematical calculations.

        Supports two modes: expression evaluation and named operations.
        In expression mode, evaluates a mathematical expression string
        using a safe subset of math functions. In operation mode, applies
        a named operation to a list of numeric operands.

        Args:
            expression: Mathematical expression to evaluate. Supports
                basic arithmetic (+, -, *, /, **, %) and the following
                functions: sin, cos, tan, log, sqrt, abs, pow, exp.
                Mutually exclusive with ``operation``/``operands``.
            operation: Named operation to perform on ``operands``. Options:
                - "add": Sum of all operands.
                - "multiply": Product of all operands.
                - "mean": Arithmetic mean.
                - "median": Median value.
                - "mode": Most common value.
                - "stdev": Sample standard deviation.
                - "variance": Sample variance.
                - "min": Minimum value.
                - "max": Maximum value.
                - "range": Difference between max and min.
                - "sum_of_squares": Sum of squared values.
                - "root_mean_square": Root mean square.
                - "geometric_mean": Geometric mean (positive numbers only).
                - "harmonic_mean": Harmonic mean.
            operands: List of numbers for the named ``operation``.
            precision: Decimal precision for calculations. Defaults to 10.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                For expression mode:
                    - expression (str): The original expression.
                    - result (float): Computed result.
                    - decimal_result (str): High-precision decimal representation.
                For operation mode:
                    - operation (str): The operation performed.
                    - operands (list[float]): The input operands.
                    - result (float): Computed result.
                    - count (int): Number of operands.
                    - note (str): Additional info (e.g., for mode with no unique value).
                - error (str): Error message if the calculation failed.

        Example:
            >>> result = Calculator.static_call(expression="sqrt(16) + pow(2, 3)")
            >>> print(result["result"])
            12.0
        """
        result = {}
        getcontext().prec = precision

        if expression:
            try:
                allowed_funcs = ["sin", "cos", "tan", "log", "sqrt", "abs", "pow", "exp"]

                safe_expr = expression
                for func in allowed_funcs:
                    safe_expr = safe_expr.replace(func, f"math.{func}")

                safe_dict = {"__builtins__": {}, "math": math}
                value = eval(safe_expr, safe_dict)

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
                    product = 1
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
    """Statistical analysis and descriptive statistics tool.

    Provides comprehensive statistical analysis including descriptive
    statistics, distribution analysis, and correlation calculations.
    Automatically calculates quartiles, outliers, and confidence intervals.

    Supported analysis types:
        descriptive: Mean, median, mode, std dev, quartiles, outliers.
        distribution: Skewness, kurtosis, frequency distribution.
        correlation: Pearson correlation for paired data.
    """

    @staticmethod
    def static_call(
        data: list[float],
        analysis_type: str = "descriptive",
        confidence_level: float = 0.95,
        **context_variables,
    ) -> dict[str, Any]:
        """Perform statistical analysis on numerical data.

        Computes comprehensive statistics based on the selected analysis
        type. Supports descriptive statistics with outlier detection,
        distribution analysis with skewness and kurtosis, and Pearson
        correlation for paired data.

        Args:
            data: List of numerical values to analyze. Must not be empty.
                For "correlation" analysis, must have an even number of
                elements (first half = x values, second half = y values).
            analysis_type: Type of statistical analysis. Options:
                - "descriptive": Mean, median, mode, std dev, variance,
                  range, sum, quartiles (Q1/Q2/Q3/IQR), and outlier
                  detection using the 1.5*IQR method.
                - "distribution": Skewness, kurtosis (excess), and
                  frequency distribution histogram with configurable bins.
                  Requires at least 3 data points.
                - "correlation": Pearson correlation coefficient for paired
                  data. Returns r, r-squared, strength label, and direction.
                  Requires an even number of data points.
            confidence_level: Confidence level for statistical intervals.
                Defaults to 0.95. Currently reserved for future use.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                - data_points (int): Number of data values.
                For "descriptive":
                    - statistics: count, mean, median, min, max, range, sum,
                      std_dev, variance, mode.
                    - quartiles: Q1, Q2, Q3, IQR.
                    - outliers: count, values, lower_bound, upper_bound.
                For "distribution":
                    - skewness (float): Measure of distribution asymmetry.
                    - kurtosis (float): Excess kurtosis.
                    - frequency_distribution: List of bin dicts with range,
                      count, and frequency.
                For "correlation":
                    - correlation: pearson_r, r_squared, strength, direction.
                - error (str): Error message if the analysis failed.

        Example:
            >>> result = StatisticalAnalyzer.static_call([1, 2, 3, 4, 5])
            >>> print(result["statistics"]["mean"])
            3
        """
        if not data:
            return {"error": "Data cannot be empty"}

        result = {"data_points": len(data)}

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
    """Advanced mathematical functions and operations.

    Provides a wide range of mathematical function evaluations
    including trigonometric, logarithmic, exponential, and rounding
    functions. Supports additional parameters for certain functions.

    Supported functions:
        Trigonometric: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh.
        Logarithmic: log (with custom base), log10.
        Exponential: exp, pow (with custom exponent).
        Rounding: floor, ceil, round (with decimal places).
        Other: sqrt, abs, factorial.
    """

    @staticmethod
    def static_call(
        function: str,
        input_value: float | None = None,
        parameters: dict[str, float] | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Evaluate a mathematical function on a single input value.

        Computes the result of the specified mathematical function applied
        to the input value. Some functions accept additional parameters
        via the ``parameters`` dictionary.

        Args:
            function: Name of the mathematical function to evaluate. Options:
                - Trigonometric: "sin", "cos", "tan", "asin", "acos", "atan",
                  "sinh", "cosh", "tanh".
                - Logarithmic: "log" (natural log; use parameters["base"]
                  for custom base), "log10".
                - Exponential: "exp", "pow" (uses parameters["exponent"],
                  default 2).
                - Rounding: "floor", "ceil", "round" (uses
                  parameters["decimals"], default 0).
                - Other: "sqrt", "abs", "factorial" (non-negative integers only).
            input_value: The numeric input to the function. Required for all
                functions. Must satisfy domain constraints (e.g., positive
                for log, [-1, 1] for asin/acos, non-negative for sqrt,
                non-negative integer for factorial).
            parameters: Additional parameters for functions that need them:
                - "base" (float): Logarithm base for "log" (default: e).
                - "exponent" (float): Exponent for "pow" (default: 2).
                - "decimals" (int): Decimal places for "round" (default: 0).
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                - function (str): The function that was evaluated.
                - input (float): The input value.
                - result (float|int): The computed result.
                - parameters (dict): Additional parameters used (if any).
                - error (str): Error message if evaluation failed.

        Example:
            >>> result = MathematicalFunctions.static_call("log", 100, {"base": 10})
            >>> print(result["result"])
            2.0
        """
        result = {}

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
    """Number theory and discrete mathematics functions.

    Provides operations for prime number checking, factorization,
    greatest common divisor, least common multiple, and mathematical
    sequences like Fibonacci and Collatz.

    Supported operations:
        prime: Check if a number is prime.
        factors: Get all factors and prime factorization.
        gcd: Calculate greatest common divisor of multiple numbers.
        lcm: Calculate least common multiple of multiple numbers.
        fibonacci: Generate Fibonacci sequence up to n terms.
        collatz: Generate Collatz sequence starting from n.
    """

    @staticmethod
    def static_call(
        operation: str,
        number: int | None = None,
        numbers: list[int] | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Perform number theory and discrete mathematics operations.

        Provides primality testing, factorization, GCD/LCM computation,
        and mathematical sequence generation.

        Args:
            operation: The number theory operation to perform. Options:
                - "prime": Check if ``number`` is prime. Returns the
                  primality status and classification (prime/composite/neither).
                - "factors": Get all factors and the prime factorization
                  of ``number``.
                - "gcd": Compute the greatest common divisor of two or
                  more ``numbers``.
                - "lcm": Compute the least common multiple of two or
                  more ``numbers``.
                - "fibonacci": Generate the first ``number`` terms of the
                  Fibonacci sequence.
                - "collatz": Generate the Collatz sequence starting from
                  ``number`` (capped at 1000 steps for safety).
            number: A single integer for operations that require one number
                (prime, factors, fibonacci, collatz).
            numbers: A list of integers for operations that require multiple
                numbers (gcd, lcm). Must contain at least 2 elements.
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing operation-specific results:
                For "prime": number, is_prime (bool), type (str).
                For "factors": number, factors (list), prime_factors (list),
                    factor_count (int).
                For "gcd": numbers (list), gcd (int).
                For "lcm": numbers (list), lcm (int).
                For "fibonacci": length, sequence (list), nth_fibonacci (int).
                For "collatz": starting_number, sequence (list), steps (int),
                    max_value (int).
                - error (str): Error message if the operation failed.

        Example:
            >>> result = NumberTheory.static_call("prime", number=17)
            >>> print(result["is_prime"])
            True
        """
        result = {}

        if operation == "prime":
            if number is None:
                return {"error": "number required for prime check"}

            def is_prime(n):
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
                factors = []
                for i in range(1, int(math.sqrt(abs(n))) + 1):
                    if n % i == 0:
                        factors.append(i)
                        if i != n // i:
                            factors.append(n // i)
                return sorted(factors)

            def prime_factors(n):
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
                while b:
                    a, b = b, a % b
                return a

            def gcd_multiple(nums):
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
                while b:
                    a, b = b, a % b
                return a

            def lcm(a, b):
                return abs(a * b) // gcd(a, b)

            def lcm_multiple(nums):
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
    """Convert between different units of measurement.

    Provides conversion between units across multiple measurement
    categories. Automatically detects the category based on units
    when not explicitly specified. Special handling for temperature.

    Supported categories:
        length: meter, centimeter, millimeter, kilometer, inch, foot, yard, mile.
        weight: gram, kilogram, pound, ounce, stone, ton.
        volume: liter, milliliter, gallon, quart, pint, cup, fluid_ounce.
        area: square_meter, square_centimeter, square_kilometer, square_foot, acre, hectare.
        speed: meter_per_second, kilometer_per_hour, mile_per_hour, knot.
        temperature: celsius, fahrenheit, kelvin.
    """

    @staticmethod
    def static_call(
        value: float,
        from_unit: str,
        to_unit: str,
        category: str | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Convert a value between different units of measurement.

        Performs unit conversion using predefined conversion factors.
        Automatically detects the measurement category when not specified.
        Temperature conversions are handled with dedicated formulas rather
        than simple multiplication.

        Args:
            value: The numeric value to convert.
            from_unit: Source unit name or abbreviation. Accepts both full
                names (e.g., "kilometer") and abbreviations (e.g., "km").
                Case-insensitive.
            to_unit: Target unit name or abbreviation. Same format as
                ``from_unit``.
            category: Measurement category to use. If None, the category
                is auto-detected from the unit names. Options:
                - "length": meter, cm, mm, km, inch, foot, yard, mile.
                - "weight": gram, kg, pound, ounce, stone, ton.
                - "volume": liter, ml, gallon, quart, pint, cup, fluid_ounce.
                - "area": square_meter, cm2, km2, square_foot, acre, hectare.
                - "speed": meter_per_second, kmh, mph, knot.
                - "temperature": celsius, fahrenheit, kelvin (auto-detected).
            **context_variables: Runtime context from the agent (unused).

        Returns:
            A dictionary containing:
                - value (float): The original input value.
                - from_unit (str): The source unit.
                - to_unit (str): The target unit.
                - result (float): The converted value.
                - category (str): The measurement category used.
                - error (str): Error message if the conversion failed.

        Example:
            >>> result = UnitConverter.static_call(100, "celsius", "fahrenheit")
            >>> print(result["result"])
            212.0
        """
        result = {}

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
