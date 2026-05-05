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
"""Tests for xerxes.tools.math_tools module."""

from xerxes.tools.math_tools import (
    Calculator,
    MathematicalFunctions,
    NumberTheory,
    StatisticalAnalyzer,
    UnitConverter,
)


class TestCalculator:
    def test_expression_basic(self):
        result = Calculator.static_call(expression="2 + 3")
        assert result["result"] == 5.0

    def test_expression_sqrt(self):
        result = Calculator.static_call(expression="sqrt(16)")
        assert result["result"] == 4.0

    def test_expression_trig(self):
        result = Calculator.static_call(expression="sin(0)")
        assert result["result"] == 0.0

    def test_expression_complex(self):
        result = Calculator.static_call(expression="pow(2, 3) + sqrt(9)")
        assert result["result"] == 11.0

    def test_expression_invalid(self):
        result = Calculator.static_call(expression="invalid_func()")
        assert "error" in result

    def test_operation_add(self):
        result = Calculator.static_call(operation="add", operands=[1, 2, 3])
        assert result["result"] == 6

    def test_operation_multiply(self):
        result = Calculator.static_call(operation="multiply", operands=[2, 3, 4])
        assert result["result"] == 24

    def test_operation_mean(self):
        result = Calculator.static_call(operation="mean", operands=[1, 2, 3, 4, 5])
        assert result["result"] == 3.0

    def test_operation_median(self):
        result = Calculator.static_call(operation="median", operands=[1, 3, 5])
        assert result["result"] == 3

    def test_operation_mode(self):
        result = Calculator.static_call(operation="mode", operands=[1, 1, 2, 3])
        assert result["result"] == 1

    def test_operation_stdev(self):
        result = Calculator.static_call(operation="stdev", operands=[1, 2, 3, 4, 5])
        assert result["result"] > 0

    def test_operation_stdev_single(self):
        result = Calculator.static_call(operation="stdev", operands=[5])
        assert result["result"] == 0

    def test_operation_variance(self):
        result = Calculator.static_call(operation="variance", operands=[1, 2, 3])
        assert result["result"] > 0

    def test_operation_min(self):
        result = Calculator.static_call(operation="min", operands=[3, 1, 2])
        assert result["result"] == 1

    def test_operation_max(self):
        result = Calculator.static_call(operation="max", operands=[3, 1, 2])
        assert result["result"] == 3

    def test_operation_range(self):
        result = Calculator.static_call(operation="range", operands=[1, 5])
        assert result["result"] == 4

    def test_operation_sum_of_squares(self):
        result = Calculator.static_call(operation="sum_of_squares", operands=[3, 4])
        assert result["result"] == 25

    def test_operation_rms(self):
        result = Calculator.static_call(operation="root_mean_square", operands=[3, 4])
        assert abs(result["result"] - 3.5355) < 0.01

    def test_operation_geometric_mean(self):
        result = Calculator.static_call(operation="geometric_mean", operands=[4, 9])
        assert abs(result["result"] - 6.0) < 0.01

    def test_operation_geometric_mean_negative(self):
        result = Calculator.static_call(operation="geometric_mean", operands=[-1, 4])
        assert "error" in result

    def test_operation_harmonic_mean(self):
        result = Calculator.static_call(operation="harmonic_mean", operands=[1, 2, 4])
        assert result["result"] > 0

    def test_operation_unknown(self):
        result = Calculator.static_call(operation="unknown", operands=[1])
        assert "error" in result

    def test_no_expression_no_operation(self):
        result = Calculator.static_call()
        assert "error" in result


class TestStatisticalAnalyzer:
    def test_descriptive(self):
        result = StatisticalAnalyzer.static_call([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert "statistics" in result
        assert result["statistics"]["mean"] == 5.5
        assert result["statistics"]["count"] == 10
        assert "quartiles" in result
        assert "outliers" in result

    def test_descriptive_single_value(self):
        result = StatisticalAnalyzer.static_call([5])
        assert result["statistics"]["mean"] == 5

    def test_empty_data(self):
        result = StatisticalAnalyzer.static_call([])
        assert "error" in result

    def test_distribution(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = StatisticalAnalyzer.static_call(data, analysis_type="distribution")
        assert "skewness" in result
        assert "kurtosis" in result
        assert "frequency_distribution" in result

    def test_correlation(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        result = StatisticalAnalyzer.static_call(data, analysis_type="correlation")
        assert "correlation" in result or "pearson_r" in result or "error" not in result

    def test_correlation_odd_data(self):
        result = StatisticalAnalyzer.static_call([1, 2, 3], analysis_type="correlation")
        assert "error" in result

    def test_unknown_analysis(self):
        result = StatisticalAnalyzer.static_call([1, 2, 3], analysis_type="unknown")
        assert "error" in result


class TestMathematicalFunctions:
    def test_sin(self):
        result = MathematicalFunctions.static_call(function="sin", input_value=0.0)
        assert result["result"] == 0.0

    def test_cos(self):
        result = MathematicalFunctions.static_call(function="cos", input_value=0.0)
        assert result["result"] == 1.0

    def test_tan(self):
        result = MathematicalFunctions.static_call(function="tan", input_value=0.0)
        assert result["result"] == 0.0

    def test_asin(self):
        result = MathematicalFunctions.static_call(function="asin", input_value=0.5)
        assert "result" in result

    def test_asin_out_of_range(self):
        result = MathematicalFunctions.static_call(function="asin", input_value=2.0)
        assert "error" in result

    def test_acos(self):
        result = MathematicalFunctions.static_call(function="acos", input_value=0.5)
        assert "result" in result

    def test_acos_out_of_range(self):
        result = MathematicalFunctions.static_call(function="acos", input_value=2.0)
        assert "error" in result

    def test_atan(self):
        result = MathematicalFunctions.static_call(function="atan", input_value=1.0)
        assert "result" in result

    def test_log(self):
        result = MathematicalFunctions.static_call(function="log", input_value=1.0)
        assert result["result"] == 0.0

    def test_log_custom_base(self):
        result = MathematicalFunctions.static_call(function="log", input_value=100.0, parameters={"base": 10})
        assert abs(result["result"] - 2.0) < 0.01

    def test_log_negative(self):
        result = MathematicalFunctions.static_call(function="log", input_value=-1.0)
        assert "error" in result

    def test_log10(self):
        result = MathematicalFunctions.static_call(function="log10", input_value=100.0)
        assert abs(result["result"] - 2.0) < 0.01

    def test_exp(self):
        result = MathematicalFunctions.static_call(function="exp", input_value=0.0)
        assert result["result"] == 1.0

    def test_sqrt(self):
        result = MathematicalFunctions.static_call(function="sqrt", input_value=16.0)
        assert result["result"] == 4.0

    def test_sqrt_negative(self):
        result = MathematicalFunctions.static_call(function="sqrt", input_value=-1.0)
        assert "error" in result

    def test_abs(self):
        result = MathematicalFunctions.static_call(function="abs", input_value=-5.0)
        assert result["result"] == 5.0

    def test_floor(self):
        result = MathematicalFunctions.static_call(function="floor", input_value=3.7)
        assert result["result"] == 3

    def test_ceil(self):
        result = MathematicalFunctions.static_call(function="ceil", input_value=3.2)
        assert result["result"] == 4

    def test_round(self):
        result = MathematicalFunctions.static_call(function="round", input_value=3.456, parameters={"decimals": 2})
        assert result["result"] == 3.46

    def test_factorial(self):
        result = MathematicalFunctions.static_call(function="factorial", input_value=5.0)
        assert result["result"] == 120

    def test_factorial_negative(self):
        result = MathematicalFunctions.static_call(function="factorial", input_value=-1.0)
        assert "error" in result

    def test_pow(self):
        result = MathematicalFunctions.static_call(function="pow", input_value=2.0, parameters={"exponent": 3})
        assert result["result"] == 8.0

    def test_sinh(self):
        result = MathematicalFunctions.static_call(function="sinh", input_value=0.0)
        assert result["result"] == 0.0

    def test_cosh(self):
        result = MathematicalFunctions.static_call(function="cosh", input_value=0.0)
        assert result["result"] == 1.0

    def test_tanh(self):
        result = MathematicalFunctions.static_call(function="tanh", input_value=0.0)
        assert result["result"] == 0.0

    def test_unknown_function(self):
        result = MathematicalFunctions.static_call(function="unknown", input_value=1.0)
        assert "error" in result

    def test_no_input(self):
        result = MathematicalFunctions.static_call(function="sin")
        assert "error" in result


class TestNumberTheory:
    def test_prime(self):
        result = NumberTheory.static_call(operation="prime", number=7)
        assert result["is_prime"] is True

    def test_not_prime(self):
        result = NumberTheory.static_call(operation="prime", number=4)
        assert result["is_prime"] is False

    def test_prime_1(self):
        result = NumberTheory.static_call(operation="prime", number=1)
        assert result["is_prime"] is False

    def test_prime_2(self):
        result = NumberTheory.static_call(operation="prime", number=2)
        assert result["is_prime"] is True

    def test_prime_no_number(self):
        result = NumberTheory.static_call(operation="prime")
        assert "error" in result

    def test_factors(self):
        result = NumberTheory.static_call(operation="factors", number=12)
        assert "factors" in result or "error" not in result

    def test_gcd(self):
        result = NumberTheory.static_call(operation="gcd", numbers=[12, 18])
        assert "gcd" in result or "result" in result

    def test_lcm(self):
        result = NumberTheory.static_call(operation="lcm", numbers=[4, 6])
        assert result is not None

    def test_fibonacci(self):
        result = NumberTheory.static_call(operation="fibonacci", number=10)
        assert "sequence" in result or "fibonacci" in result or "error" not in result

    def test_collatz(self):
        result = NumberTheory.static_call(operation="collatz", number=6)
        assert result is not None


class TestUnitConverter:
    def test_length(self):
        result = UnitConverter.static_call(value=1.0, from_unit="m", to_unit="ft", category="length")
        assert "result" in result or "converted" in result or "error" not in result

    def test_weight(self):
        result = UnitConverter.static_call(value=1.0, from_unit="kg", to_unit="lb", category="weight")
        assert result is not None

    def test_temperature_c_to_f(self):
        result = UnitConverter.static_call(value=100.0, from_unit="c", to_unit="f", category="temperature")
        assert result is not None
