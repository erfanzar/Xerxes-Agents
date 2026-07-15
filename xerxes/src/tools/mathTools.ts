// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'

const MAX_EXPRESSION_LENGTH = 10_000
const MAX_EXPRESSION_TOKENS = 1_000
const MAX_EXPRESSION_DEPTH = 100
const MAX_FACTORIAL_INPUT = 5_000
const MAX_SEQUENCE_LENGTH = 1_000
const MAX_FACTORIZATION_INPUT = 10_000_000_000
const MAX_PRIME_INPUT = 1_000_000_000_000

type NumericParameters = Readonly<Record<string, number>>

export const CALCULATOR_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'Calculator',
    description: 'Safely evaluate arithmetic expressions or calculate named aggregates.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        expression: {
          type: 'string',
          description: 'Arithmetic expression using +, -, *, /, //, %, **, pi, e, and approved functions.',
        },
        operation: {
          type: 'string',
          enum: [
            'add',
            'multiply',
            'mean',
            'median',
            'mode',
            'stdev',
            'variance',
            'min',
            'max',
            'range',
            'sum_of_squares',
            'root_mean_square',
            'geometric_mean',
            'harmonic_mean',
          ],
        },
        operands: { type: 'array', items: { type: 'number' } },
        precision: { type: 'integer', default: 10, description: 'Validated for compatibility with the Python tool.' },
      },
    },
  },
}

export const STATISTICAL_ANALYZER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'StatisticalAnalyzer',
    description: 'Analyze numerical data with descriptive, distribution, or paired correlation statistics.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        data: { type: 'array', items: { type: 'number' } },
        analysis_type: { type: 'string', enum: ['descriptive', 'distribution', 'correlation'], default: 'descriptive' },
        confidence_level: { type: 'number', default: 0.95 },
      },
      required: ['data'],
    },
  },
}

export const MATHEMATICAL_FUNCTIONS_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'MathematicalFunctions',
    description: 'Evaluate trigonometric, logarithmic, rounding, factorial, power, and hyperbolic functions.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        function: {
          type: 'string',
          enum: [
            'sin',
            'cos',
            'tan',
            'asin',
            'acos',
            'atan',
            'log',
            'log10',
            'exp',
            'sqrt',
            'abs',
            'floor',
            'ceil',
            'round',
            'factorial',
            'pow',
            'sinh',
            'cosh',
            'tanh',
          ],
        },
        input_value: { type: 'number' },
        parameters: {
          type: 'object',
          additionalProperties: { type: 'number' },
          description: 'Optional numeric parameters such as base, exponent, or decimals.',
        },
      },
      required: ['function'],
    },
  },
}

export const NUMBER_THEORY_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'NumberTheory',
    description: 'Check primes, factor integers, calculate GCD/LCM, or generate Fibonacci and Collatz sequences.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        operation: { type: 'string', enum: ['prime', 'factors', 'gcd', 'lcm', 'fibonacci', 'collatz'] },
        number: { type: 'integer' },
        numbers: { type: 'array', items: { type: 'integer' } },
      },
      required: ['operation'],
    },
  },
}

export const UNIT_CONVERTER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'UnitConverter',
    description: 'Convert length, weight, volume, area, speed, or temperature measurements.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        value: { type: 'number' },
        from_unit: { type: 'string' },
        to_unit: { type: 'string' },
        category: { type: 'string', enum: ['length', 'weight', 'volume', 'area', 'speed', 'temperature'] },
      },
      required: ['value', 'from_unit', 'to_unit'],
    },
  },
}

export const MATH_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  CALCULATOR_DEFINITION,
  STATISTICAL_ANALYZER_DEFINITION,
  MATHEMATICAL_FUNCTIONS_DEFINITION,
  NUMBER_THEORY_DEFINITION,
  UNIT_CONVERTER_DEFINITION,
]

/** Register Bun-native mathematical tools with the same public names as the Python runtime. */
export function registerMathTools(registry: ToolRegistry): void {
  registry.register(CALCULATOR_DEFINITION, calculator)
  registry.register(STATISTICAL_ANALYZER_DEFINITION, statisticalAnalyzer)
  registry.register(MATHEMATICAL_FUNCTIONS_DEFINITION, mathematicalFunctions)
  registry.register(NUMBER_THEORY_DEFINITION, numberTheory)
  registry.register(UNIT_CONVERTER_DEFINITION, unitConverter)
}

/** Evaluate a calculator expression or a named aggregate without executing JavaScript source. */
export function calculator(inputs: JsonObject): JsonObject {
  const expression = optionalText(inputs, 'expression')
  const operation = optionalText(inputs, 'operation')
  const operands = optionalNumberArray(inputs, 'operands')
  const precision = optionalInteger(inputs, 'precision', 10)
  if (precision < 1 || precision > 1_000) {
    throw new ValidationError('precision', 'must be between 1 and 1000', precision)
  }

  if (expression) {
    try {
      const value = safeEvaluateExpression(expression)
      return {
        expression,
        result: value,
        decimal_result: decimalResult(value),
      }
    } catch (error) {
      return failure('Invalid expression: ' + errorText(error))
    }
  }

  if (!operation || !operands || operands.length === 0) {
    return failure('Either expression or operation with operands required')
  }

  try {
    return calculateNamedOperation(operation, operands)
  } catch (error) {
    return failure('Calculation failed: ' + errorText(error))
  }
}

/** Perform the Python tool's three supported statistical analyses. */
export function statisticalAnalyzer(inputs: JsonObject): JsonObject {
  const data = requiredNumberArray(inputs, 'data')
  if (data.length === 0) {
    return failure('Data cannot be empty')
  }
  const analysisType = optionalText(inputs, 'analysis_type') ?? 'descriptive'
  const confidenceLevel = optionalFiniteNumber(inputs, 'confidence_level')
  if (confidenceLevel !== undefined && (confidenceLevel <= 0 || confidenceLevel >= 1)) {
    throw new ValidationError('confidence_level', 'must be greater than 0 and less than 1', confidenceLevel)
  }

  switch (analysisType) {
    case 'descriptive':
      return descriptiveStatistics(data)
    case 'distribution':
      return distributionStatistics(data)
    case 'correlation':
      return correlationStatistics(data)
    default:
      return failure('Unknown analysis type: ' + analysisType)
  }
}

/** Evaluate one named mathematical function. */
export function mathematicalFunctions(inputs: JsonObject): JsonObject {
  const functionName = requiredText(inputs, 'function')
  const inputValue = optionalFiniteNumber(inputs, 'input_value')
  if (inputValue === undefined) {
    return failure('input_value required')
  }
  const parameters = optionalNumericParameters(inputs, 'parameters')

  try {
    const value = evaluateNamedFunction(functionName, inputValue, parameters)
    const result: JsonObject = {
      function: functionName,
      input: inputValue,
      result: value,
    }
    if (parameters) {
      result.parameters = parameters
    }
    return result
  } catch (error) {
    return failure('Function evaluation failed: ' + errorText(error))
  }
}

/** Run one finite, bounded number-theory operation. */
export function numberTheory(inputs: JsonObject): JsonObject {
  const operation = requiredText(inputs, 'operation')
  const number = optionalSafeInteger(inputs, 'number')
  const numbers = optionalSafeIntegerArray(inputs, 'numbers')

  switch (operation) {
    case 'prime':
      if (number === undefined) {
        return failure('number required for prime check')
      }
      if (Math.abs(number) > MAX_PRIME_INPUT) {
        return failure('Prime checks are limited to absolute values at most ' + MAX_PRIME_INPUT)
      }
      return {
        number,
        is_prime: isPrime(number),
        type: isPrime(number) ? 'prime' : number > 1 ? 'composite' : 'neither',
      }
    case 'factors':
      if (number === undefined) {
        return failure('number required for factorization')
      }
      if (Math.abs(number) > MAX_FACTORIZATION_INPUT) {
        return failure('Factorization is limited to absolute values at most ' + MAX_FACTORIZATION_INPUT)
      }
      return {
        number,
        factors: factorsOf(Math.abs(number)),
        prime_factors: primeFactorsOf(Math.abs(number)),
        factor_count: factorsOf(Math.abs(number)).length,
      }
    case 'gcd':
      if (!numbers || numbers.length < 2) {
        return failure('At least 2 numbers required for GCD')
      }
      return { numbers, gcd: integerResult(gcdMany(numbers)) }
    case 'lcm':
      if (!numbers || numbers.length < 2) {
        return failure('At least 2 numbers required for LCM')
      }
      return { numbers, lcm: integerResult(lcmMany(numbers)) }
    case 'fibonacci':
      if (number === undefined) {
        return failure('number required for Fibonacci sequence')
      }
      if (number < 0 || number > MAX_SEQUENCE_LENGTH) {
        return failure('Fibonacci length must be between 0 and ' + MAX_SEQUENCE_LENGTH)
      }
      return fibonacciResult(number)
    case 'collatz':
      if (number === undefined) {
        return failure('number required for Collatz sequence')
      }
      if (number <= 0) {
        return failure('Collatz starting number must be positive')
      }
      return collatzResult(number)
    default:
      return failure('Unknown operation: ' + operation)
  }
}

/** Convert compatible units while preserving the Python tool's result shape. */
export function unitConverter(inputs: JsonObject): JsonObject {
  const value = requiredFiniteNumber(inputs, 'value')
  const fromUnit = requiredText(inputs, 'from_unit')
  const toUnit = requiredText(inputs, 'to_unit')
  const requestedCategory = optionalText(inputs, 'category')?.toLowerCase()
  const source = fromUnit.toLowerCase()
  const target = toUnit.toLowerCase()

  const sourceTemperature = TEMPERATURE_UNITS.has(source)
  const targetTemperature = TEMPERATURE_UNITS.has(target)
  if (sourceTemperature || targetTemperature || requestedCategory === 'temperature') {
    if (!sourceTemperature || !targetTemperature) {
      return failure('Temperature conversions require both units to be celsius, fahrenheit, or kelvin')
    }
    if (requestedCategory && requestedCategory !== 'temperature') {
      return failure('Temperature units cannot use category ' + requestedCategory)
    }
    return {
      value,
      from_unit: fromUnit,
      to_unit: toUnit,
      result: convertTemperature(value, source, target),
      category: 'temperature',
    }
  }

  const category = requestedCategory ?? findCategory(source, target)
  if (!category) {
    return failure('Could not determine category for units ' + fromUnit + ' and ' + toUnit)
  }
  const conversions = UNIT_CONVERSIONS[category]
  if (!conversions) {
    return failure('Unknown category: ' + category)
  }
  const sourceFactor = conversions[source]
  const targetFactor = conversions[target]
  if (sourceFactor === undefined) {
    return failure('Unknown unit: ' + fromUnit + ' in category ' + category)
  }
  if (targetFactor === undefined) {
    return failure('Unknown unit: ' + toUnit + ' in category ' + category)
  }

  return {
    value,
    from_unit: fromUnit,
    to_unit: toUnit,
    result: value * sourceFactor / targetFactor,
    category,
  }
}

/**
 * Evaluate a restricted expression grammar. It deliberately has no property access,
 * array literals, assignments, or JavaScript evaluation path.
 */
export function safeEvaluateExpression(expression: string): number {
  return new ExpressionParser(expression).parse()
}

function calculateNamedOperation(operation: string, operands: readonly number[]): JsonObject {
  let value: number
  switch (operation) {
    case 'add':
      value = sum(operands)
      break
    case 'multiply':
      value = operands.reduce((product, item) => product * item, 1)
      break
    case 'mean':
      value = mean(operands)
      break
    case 'median':
      value = median(operands)
      break
    case 'mode':
      value = mode(operands)
      break
    case 'stdev':
      value = operands.length > 1 ? Math.sqrt(sampleVariance(operands)) : 0
      break
    case 'variance':
      value = operands.length > 1 ? sampleVariance(operands) : 0
      break
    case 'min':
      value = Math.min(...operands)
      break
    case 'max':
      value = Math.max(...operands)
      break
    case 'range':
      value = Math.max(...operands) - Math.min(...operands)
      break
    case 'sum_of_squares':
      value = sum(operands.map(item => item ** 2))
      break
    case 'root_mean_square':
      value = Math.sqrt(sum(operands.map(item => item ** 2)) / operands.length)
      break
    case 'geometric_mean':
      if (operands.some(item => item <= 0)) {
        return failure('Geometric mean requires positive numbers')
      }
      value = Math.exp(sum(operands.map(Math.log)) / operands.length)
      break
    case 'harmonic_mean':
      if (operands.some(item => item < 0)) {
        return failure('Harmonic mean requires non-negative numbers')
      }
      value = operands.some(item => item === 0) ? 0 : operands.length / sum(operands.map(item => 1 / item))
      break
    default:
      return failure('Unknown operation: ' + operation)
  }
  return {
    operation,
    operands: [...operands],
    result: assertFinite(value),
    count: operands.length,
  }
}

function descriptiveStatistics(data: readonly number[]): JsonObject {
  const sorted = [...data].sort((left, right) => left - right)
  const q1 = exclusiveQuantile(sorted, 0.25)
  const q2 = exclusiveQuantile(sorted, 0.5)
  const q3 = exclusiveQuantile(sorted, 0.75)
  const iqr = q3 - q1
  const lowerBound = q1 - 1.5 * iqr
  const upperBound = q3 + 1.5 * iqr
  const outliers = data.filter(item => item < lowerBound || item > upperBound)
  const statistics: JsonObject = {
    count: data.length,
    mean: mean(data),
    median: median(sorted),
    min: sorted[0]!,
    max: sorted.at(-1)!,
    range: sorted.at(-1)! - sorted[0]!,
    sum: sum(data),
    mode: mode(data),
  }
  if (data.length > 1) {
    statistics.std_dev = Math.sqrt(sampleVariance(data))
    statistics.variance = sampleVariance(data)
  }
  return {
    data_points: data.length,
    statistics,
    quartiles: { Q1: q1, Q2: q2, Q3: q3, IQR: iqr },
    outliers: {
      count: outliers.length,
      values: outliers.slice(0, 20),
      lower_bound: lowerBound,
      upper_bound: upperBound,
    },
  }
}

function distributionStatistics(data: readonly number[]): JsonObject {
  const result: JsonObject = { data_points: data.length }
  const average = mean(data)
  if (data.length > 2) {
    const stdDev = Math.sqrt(sampleVariance(data))
    if (stdDev === 0) {
      result.skewness = 0
      result.kurtosis = 0
    } else {
      result.skewness = sum(data.map(item => (item - average) ** 3)) / (data.length * stdDev ** 3)
      result.kurtosis = sum(data.map(item => (item - average) ** 4)) / (data.length * stdDev ** 4) - 3
    }
  }

  const uniqueCount = new Set(data).size
  const binsCount = Math.min(10, uniqueCount)
  if (binsCount > 1) {
    const minimum = Math.min(...data)
    const maximum = Math.max(...data)
    const binWidth = (maximum - minimum) / binsCount
    const bins: JsonObject[] = []
    for (let index = 0; index < binsCount; index += 1) {
      const start = minimum + index * binWidth
      const end = start + binWidth
      const count = data.filter(item => (
        (item >= start && item < end) || (index === binsCount - 1 && item === end)
      )).length
      bins.push({
        range: start.toFixed(2) + ' - ' + end.toFixed(2),
        count,
        frequency: count / data.length,
      })
    }
    result.frequency_distribution = bins
  }
  return result
}

function correlationStatistics(data: readonly number[]): JsonObject {
  if (data.length % 2 !== 0) {
    return failure('Correlation analysis requires paired data (even number of values)')
  }
  const midpoint = data.length / 2
  const xData = data.slice(0, midpoint)
  const yData = data.slice(midpoint)
  const meanX = mean(xData)
  const meanY = mean(yData)
  const numerator = sum(xData.map((item, index) => (item - meanX) * (yData[index]! - meanY)))
  const sumSquaresX = sum(xData.map(item => (item - meanX) ** 2))
  const sumSquaresY = sum(yData.map(item => (item - meanY) ** 2))
  if (sumSquaresX * sumSquaresY === 0) {
    return {
      data_points: data.length,
      correlation: { error: 'Cannot calculate correlation (zero variance)' },
    }
  }
  const correlation = numerator / Math.sqrt(sumSquaresX * sumSquaresY)
  return {
    data_points: data.length,
    correlation: {
      pearson_r: correlation,
      r_squared: correlation ** 2,
      strength: Math.abs(correlation) > 0.7 ? 'strong' : Math.abs(correlation) > 0.4 ? 'moderate' : 'weak',
      direction: correlation > 0 ? 'positive' : correlation < 0 ? 'negative' : 'none',
    },
  }
}

function evaluateNamedFunction(functionName: string, inputValue: number, parameters?: NumericParameters): JsonValue {
  switch (functionName) {
    case 'sin':
      return assertFinite(Math.sin(inputValue))
    case 'cos':
      return assertFinite(Math.cos(inputValue))
    case 'tan':
      return assertFinite(Math.tan(inputValue))
    case 'asin':
      if (inputValue < -1 || inputValue > 1) {
        throw new RangeError('asin input must be between -1 and 1')
      }
      return Math.asin(inputValue)
    case 'acos':
      if (inputValue < -1 || inputValue > 1) {
        throw new RangeError('acos input must be between -1 and 1')
      }
      return Math.acos(inputValue)
    case 'atan':
      return Math.atan(inputValue)
    case 'log': {
      if (inputValue <= 0) {
        throw new RangeError('log input must be positive')
      }
      const base = parameters?.base ?? Math.E
      if (base <= 0 || base === 1) {
        throw new RangeError('log base must be positive and not equal to 1')
      }
      return assertFinite(Math.log(inputValue) / Math.log(base))
    }
    case 'log10':
      if (inputValue <= 0) {
        throw new RangeError('log10 input must be positive')
      }
      return Math.log10(inputValue)
    case 'exp':
      return assertFinite(Math.exp(inputValue))
    case 'sqrt':
      if (inputValue < 0) {
        throw new RangeError('sqrt input must be non-negative')
      }
      return Math.sqrt(inputValue)
    case 'abs':
      return Math.abs(inputValue)
    case 'floor':
      return Math.floor(inputValue)
    case 'ceil':
      return Math.ceil(inputValue)
    case 'round': {
      const decimals = parameters?.decimals ?? 0
      if (!Number.isInteger(decimals) || decimals < -15 || decimals > 15) {
        throw new RangeError('round decimals must be an integer between -15 and 15')
      }
      return roundHalfEven(inputValue, decimals)
    }
    case 'factorial':
      if (!Number.isSafeInteger(inputValue) || inputValue < 0) {
        throw new RangeError('factorial input must be non-negative integer')
      }
      if (inputValue > MAX_FACTORIAL_INPUT) {
        throw new RangeError('factorial input must be at most ' + MAX_FACTORIAL_INPUT)
      }
      return factorial(inputValue)
    case 'pow': {
      const exponent = parameters?.exponent ?? 2
      return boundedPower(inputValue, exponent)
    }
    case 'sinh':
      return assertFinite(Math.sinh(inputValue))
    case 'cosh':
      return assertFinite(Math.cosh(inputValue))
    case 'tanh':
      return Math.tanh(inputValue)
    default:
      throw new RangeError('Unknown function: ' + functionName)
  }
}

function fibonacciResult(length: number): JsonObject {
  const sequence: JsonValue[] = []
  let previous = 0n
  let current = 1n
  for (let index = 0; index < length; index += 1) {
    sequence.push(integerResult(previous))
    const next = previous + current
    previous = current
    current = next
  }
  const result: JsonObject = { length, sequence }
  if (length > 0) {
    result.nth_fibonacci = sequence.at(-1)!
  }
  return result
}

function collatzResult(startingNumber: number): JsonObject {
  const sequence: bigint[] = [BigInt(startingNumber)]
  let current = sequence[0]!
  while (current !== 1n && sequence.length <= 1_000) {
    current = current % 2n === 0n ? current / 2n : current * 3n + 1n
    sequence.push(current)
  }
  const maxValue = sequence.reduce((largest, value) => value > largest ? value : largest)
  return {
    starting_number: startingNumber,
    sequence: sequence.map(integerResult),
    steps: sequence.length - 1,
    max_value: integerResult(maxValue),
  }
}

function isPrime(value: number): boolean {
  if (value < 2) {
    return false
  }
  if (value === 2) {
    return true
  }
  if (value % 2 === 0) {
    return false
  }
  for (let divisor = 3; divisor * divisor <= value; divisor += 2) {
    if (value % divisor === 0) {
      return false
    }
  }
  return true
}

function factorsOf(value: number): number[] {
  if (value === 0) {
    return []
  }
  const factors: number[] = []
  for (let divisor = 1; divisor * divisor <= value; divisor += 1) {
    if (value % divisor === 0) {
      factors.push(divisor)
      const complement = value / divisor
      if (complement !== divisor) {
        factors.push(complement)
      }
    }
  }
  return factors.sort((left, right) => left - right)
}

function primeFactorsOf(value: number): number[] {
  const factors: number[] = []
  let remaining = value
  for (let divisor = 2; divisor * divisor <= remaining; divisor += 1) {
    while (remaining % divisor === 0) {
      factors.push(divisor)
      remaining /= divisor
    }
  }
  if (remaining > 1) {
    factors.push(remaining)
  }
  return factors
}

function gcdMany(numbers: readonly number[]): bigint {
  return numbers.map(BigInt).reduce((left, right) => gcd(left, right))
}

function lcmMany(numbers: readonly number[]): bigint {
  return numbers.map(BigInt).reduce((left, right) => {
    if (left === 0n || right === 0n) {
      return 0n
    }
    return absBigInt(left * right) / gcd(left, right)
  })
}

function gcd(left: bigint, right: bigint): bigint {
  let a = absBigInt(left)
  let b = absBigInt(right)
  while (b !== 0n) {
    const next = a % b
    a = b
    b = next
  }
  return a
}

function absBigInt(value: bigint): bigint {
  return value < 0n ? -value : value
}

function integerResult(value: bigint): number | string {
  const asNumber = Number(value)
  return Number.isSafeInteger(asNumber) ? asNumber : value.toString()
}

function factorial(value: number): number | string {
  let result = 1n
  for (let current = 2n; current <= BigInt(value); current += 1n) {
    result *= current
  }
  return integerResult(result)
}

function boundedPower(base: number, exponent: number): number {
  if (base !== 0 && Math.abs(exponent) * Math.log10(Math.max(Math.abs(base), 1)) > 308) {
    throw new RangeError('Exponent too large')
  }
  return assertFinite(Math.pow(base, exponent))
}

function exclusiveQuantile(sorted: readonly number[], quantile: number): number {
  if (sorted.length === 1) {
    return sorted[0]!
  }
  const position = quantile * (sorted.length + 1)
  const lowerIndex = Math.floor(position) - 1
  const fraction = position - Math.floor(position)
  if (lowerIndex < 0) {
    return sorted[0]! + position * (sorted[1]! - sorted[0]!)
  }
  if (lowerIndex >= sorted.length - 1) {
    const final = sorted.at(-1)!
    return final + (position - sorted.length) * (final - sorted.at(-2)!)
  }
  return sorted[lowerIndex]! + fraction * (sorted[lowerIndex + 1]! - sorted[lowerIndex]!)
}

function mean(values: readonly number[]): number {
  return sum(values) / values.length
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right)
  const midpoint = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[midpoint - 1]! + sorted[midpoint]!) / 2 : sorted[midpoint]!
}

function mode(values: readonly number[]): number {
  const counts = new Map<number, number>()
  let mostFrequent = values[0]!
  let maximumCount = 0
  for (const value of values) {
    const count = (counts.get(value) ?? 0) + 1
    counts.set(value, count)
    if (count > maximumCount) {
      maximumCount = count
      mostFrequent = value
    }
  }
  return mostFrequent
}

function sampleVariance(values: readonly number[]): number {
  const average = mean(values)
  return sum(values.map(value => (value - average) ** 2)) / (values.length - 1)
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0)
}

function roundHalfEven(value: number, decimals: number): number {
  const factor = 10 ** decimals
  const scaled = value * factor
  const lower = Math.floor(scaled)
  const difference = scaled - lower
  const epsilon = Number.EPSILON * Math.max(1, Math.abs(scaled)) * 4
  let rounded: number
  if (Math.abs(difference - 0.5) <= epsilon) {
    rounded = lower % 2 === 0 ? lower : lower + 1
  } else {
    rounded = Math.round(scaled)
  }
  return rounded / factor
}

function assertFinite(value: number): number {
  if (!Number.isFinite(value)) {
    throw new RangeError('result is not finite')
  }
  return value
}

function decimalResult(value: number): string {
  return Object.is(value, -0) ? '-0' : String(value)
}

function failure(message: string): JsonObject {
  return { error: message }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function requiredText(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string' || !value) {
    throw new ValidationError(name, 'must be a non-empty string', value)
  }
  return value
}

function optionalText(inputs: JsonObject, name: string): string | undefined {
  const value = inputs[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'string') {
    throw new ValidationError(name, 'must be a string', value)
  }
  return value
}

function requiredFiniteNumber(inputs: JsonObject, name: string): number {
  const value = inputs[name]
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(name, 'must be a finite number', value)
  }
  return value
}

function optionalFiniteNumber(inputs: JsonObject, name: string): number | undefined {
  const value = inputs[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(name, 'must be a finite number', value)
  }
  return value
}

function optionalInteger(inputs: JsonObject, name: string, defaultValue: number): number {
  const value = inputs[name]
  if (value === undefined) {
    return defaultValue
  }
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new ValidationError(name, 'must be an integer', value)
  }
  return value
}

function requiredNumberArray(inputs: JsonObject, name: string): number[] {
  const value = inputs[name]
  if (!Array.isArray(value) || value.some(item => typeof item !== 'number' || !Number.isFinite(item))) {
    throw new ValidationError(name, 'must be an array of finite numbers', value)
  }
  return [...value] as number[]
}

function optionalNumberArray(inputs: JsonObject, name: string): number[] | undefined {
  if (inputs[name] === undefined) {
    return undefined
  }
  return requiredNumberArray(inputs, name)
}

function optionalSafeInteger(inputs: JsonObject, name: string): number | undefined {
  const value = inputs[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
    throw new ValidationError(name, 'must be a safe integer', value)
  }
  return value
}

function optionalSafeIntegerArray(inputs: JsonObject, name: string): number[] | undefined {
  const value = inputs[name]
  if (value === undefined) {
    return undefined
  }
  if (!Array.isArray(value) || value.some(item => typeof item !== 'number' || !Number.isSafeInteger(item))) {
    throw new ValidationError(name, 'must be an array of safe integers', value)
  }
  return [...value] as number[]
}

function optionalNumericParameters(inputs: JsonObject, name: string): NumericParameters | undefined {
  const value = inputs[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ValidationError(name, 'must be an object with finite numeric values', value)
  }
  const parameters: Record<string, number> = {}
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== 'number' || !Number.isFinite(item)) {
      throw new ValidationError(name, 'must be an object with finite numeric values', value)
    }
    parameters[key] = item
  }
  return parameters
}

type TokenKind = 'comma' | 'identifier' | 'left_parenthesis' | 'number' | 'operator' | 'right_parenthesis' | 'end'

interface ExpressionToken {
  readonly kind: TokenKind
  readonly position: number
  readonly value?: string
}

class ExpressionParser {
  private readonly tokens: readonly ExpressionToken[]
  private index = 0
  private depth = 0

  constructor(source: string) {
    if (!source.trim()) {
      throw new SyntaxError('expression must not be empty')
    }
    if (source.length > MAX_EXPRESSION_LENGTH) {
      throw new RangeError('expression exceeds ' + MAX_EXPRESSION_LENGTH + ' characters')
    }
    this.tokens = tokenize(source)
  }

  parse(): number {
    const result = this.parseAdditive()
    if (this.peek().kind !== 'end') {
      throw this.syntax('unexpected token ' + tokenLabel(this.peek()))
    }
    return assertFinite(result)
  }

  private parseAdditive(): number {
    let result = this.parseMultiplicative()
    while (this.peekOperator('+') || this.peekOperator('-')) {
      const operator = this.consume().value!
      const right = this.parseMultiplicative()
      result = operator === '+' ? result + right : result - right
      result = assertFinite(result)
    }
    return result
  }

  private parseMultiplicative(): number {
    let result = this.parseUnary()
    while (this.peekOperator('*') || this.peekOperator('/') || this.peekOperator('//') || this.peekOperator('%')) {
      const operator = this.consume().value!
      const right = this.parseUnary()
      switch (operator) {
        case '*':
          result *= right
          break
        case '/':
          result /= right
          break
        case '//':
          result = Math.floor(result / right)
          break
        case '%':
          result %= right
          break
        default:
          throw this.syntax('unsupported operator ' + operator)
      }
      result = assertFinite(result)
    }
    return result
  }

  private parseUnary(): number {
    if (this.peekOperator('+')) {
      this.consume()
      return this.parseUnary()
    }
    if (this.peekOperator('-')) {
      this.consume()
      return -this.parseUnary()
    }
    return this.parsePower()
  }

  private parsePower(): number {
    const base = this.parsePrimary()
    if (!this.peekOperator('**')) {
      return base
    }
    this.consume()
    return boundedPower(base, this.parseUnary())
  }

  private parsePrimary(): number {
    const token = this.consume()
    if (token.kind === 'number') {
      return Number(token.value)
    }
    if (token.kind === 'identifier') {
      if (this.peek().kind === 'left_parenthesis') {
        return this.parseFunctionCall(token.value!)
      }
      const constant = EXPRESSION_CONSTANTS[token.value!]
      if (constant === undefined) {
        throw this.syntax('Name ' + JSON.stringify(token.value) + ' is not allowed')
      }
      return constant
    }
    if (token.kind === 'left_parenthesis') {
      this.enterDepth()
      const result = this.parseAdditive()
      this.expect('right_parenthesis')
      this.depth -= 1
      return result
    }
    throw this.syntax('expected a number, name, or parenthesized expression')
  }

  private parseFunctionCall(functionName: string): number {
    const functionDefinition = EXPRESSION_FUNCTIONS[functionName]
    if (!functionDefinition) {
      throw this.syntax('Function ' + JSON.stringify(functionName) + ' is not allowed')
    }
    this.expect('left_parenthesis')
    this.enterDepth()
    const arguments_: number[] = []
    if (this.peek().kind !== 'right_parenthesis') {
      arguments_.push(this.parseAdditive())
      while (this.peek().kind === 'comma') {
        this.consume()
        arguments_.push(this.parseAdditive())
      }
    }
    this.expect('right_parenthesis')
    this.depth -= 1
    if (!functionDefinition.accepts(arguments_.length)) {
      throw this.syntax(functionName + ' received an invalid number of arguments')
    }
    return assertFinite(functionDefinition.call(arguments_))
  }

  private enterDepth(): void {
    this.depth += 1
    if (this.depth > MAX_EXPRESSION_DEPTH) {
      throw this.syntax('expression nesting exceeds ' + MAX_EXPRESSION_DEPTH)
    }
  }

  private expect(kind: TokenKind): ExpressionToken {
    const token = this.consume()
    if (token.kind !== kind) {
      throw this.syntax('expected ' + kind + ', found ' + tokenLabel(token))
    }
    return token
  }

  private peek(): ExpressionToken {
    return this.tokens[this.index]!
  }

  private consume(): ExpressionToken {
    const token = this.peek()
    if (token.kind === 'end') {
      throw this.syntax('unexpected end of expression')
    }
    this.index += 1
    return token
  }

  private peekOperator(value: string): boolean {
    const token = this.peek()
    return token.kind === 'operator' && token.value === value
  }

  private syntax(message: string): SyntaxError {
    return new SyntaxError(message + ' at character ' + this.peek().position)
  }
}

interface ExpressionFunction {
  readonly accepts: (count: number) => boolean
  readonly call: (arguments_: readonly number[]) => number
}

const EXPRESSION_CONSTANTS: Readonly<Record<string, number>> = {
  pi: Math.PI,
  e: Math.E,
}

const EXPRESSION_FUNCTIONS: Readonly<Record<string, ExpressionFunction>> = {
  sin: unary(Math.sin),
  cos: unary(Math.cos),
  tan: unary(Math.tan),
  log: unary(Math.log),
  sqrt: unary(Math.sqrt),
  abs: unary(Math.abs),
  exp: unary(Math.exp),
  pow: {
    accepts: count => count === 2,
    call: arguments_ => boundedPower(arguments_[0]!, arguments_[1]!),
  },
}

function unary(call: (value: number) => number): ExpressionFunction {
  return {
    accepts: count => count === 1,
    call: arguments_ => call(arguments_[0]!),
  }
}

function tokenize(source: string): ExpressionToken[] {
  const tokens: ExpressionToken[] = []
  let position = 0
  while (position < source.length) {
    const character = source[position]!
    if (/\s/.test(character)) {
      position += 1
      continue
    }
    const number = source.slice(position).match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/)
    if (number) {
      const value = Number(number[0])
      if (!Number.isFinite(value)) {
        throw new SyntaxError('numeric literal must be finite at character ' + position)
      }
      tokens.push({ kind: 'number', position, value: number[0] })
      position += number[0].length
    } else {
      const identifier = source.slice(position).match(/^[A-Za-z_][A-Za-z0-9_]*/)
      if (identifier) {
        tokens.push({ kind: 'identifier', position, value: identifier[0] })
        position += identifier[0].length
      } else if (source.startsWith('**', position) || source.startsWith('//', position)) {
        tokens.push({ kind: 'operator', position, value: source.slice(position, position + 2) })
        position += 2
      } else if ('+-*/%'.includes(character)) {
        tokens.push({ kind: 'operator', position, value: character })
        position += 1
      } else if (character === '(') {
        tokens.push({ kind: 'left_parenthesis', position })
        position += 1
      } else if (character === ')') {
        tokens.push({ kind: 'right_parenthesis', position })
        position += 1
      } else if (character === ',') {
        tokens.push({ kind: 'comma', position })
        position += 1
      } else {
        throw new SyntaxError('unsupported character ' + JSON.stringify(character) + ' at character ' + position)
      }
    }
    if (tokens.length > MAX_EXPRESSION_TOKENS) {
      throw new RangeError('expression exceeds ' + MAX_EXPRESSION_TOKENS + ' tokens')
    }
  }
  tokens.push({ kind: 'end', position })
  return tokens
}

function tokenLabel(token: ExpressionToken): string {
  return token.value ? JSON.stringify(token.value) : token.kind
}

const TEMPERATURE_UNITS = new Set(['celsius', 'c', 'fahrenheit', 'f', 'kelvin', 'k'])

const UNIT_CONVERSIONS: Readonly<Record<string, Readonly<Record<string, number>>>> = {
  length: {
    meter: 1,
    meters: 1,
    m: 1,
    centimeter: 0.01,
    centimeters: 0.01,
    cm: 0.01,
    millimeter: 0.001,
    millimeters: 0.001,
    mm: 0.001,
    kilometer: 1_000,
    kilometers: 1_000,
    km: 1_000,
    inch: 0.0254,
    inches: 0.0254,
    in: 0.0254,
    foot: 0.3048,
    feet: 0.3048,
    ft: 0.3048,
    yard: 0.9144,
    yards: 0.9144,
    yd: 0.9144,
    mile: 1_609.344,
    miles: 1_609.344,
    mi: 1_609.344,
  },
  weight: {
    gram: 1,
    grams: 1,
    g: 1,
    kilogram: 1_000,
    kilograms: 1_000,
    kg: 1_000,
    pound: 453.592,
    pounds: 453.592,
    lb: 453.592,
    ounce: 28.3495,
    ounces: 28.3495,
    oz: 28.3495,
    stone: 6_350.29,
    ton: 1_000_000,
    tonne: 1_000_000,
  },
  volume: {
    liter: 1,
    liters: 1,
    litre: 1,
    litres: 1,
    l: 1,
    milliliter: 0.001,
    milliliters: 0.001,
    millilitre: 0.001,
    millilitres: 0.001,
    ml: 0.001,
    gallon: 3.78541,
    gallons: 3.78541,
    gal: 3.78541,
    quart: 0.946353,
    quarts: 0.946353,
    qt: 0.946353,
    pint: 0.473176,
    pints: 0.473176,
    pt: 0.473176,
    cup: 0.236588,
    cups: 0.236588,
    fluid_ounce: 0.0295735,
    fl_oz: 0.0295735,
  },
  area: {
    square_meter: 1,
    square_meters: 1,
    m2: 1,
    square_centimeter: 0.0001,
    square_centimeters: 0.0001,
    cm2: 0.0001,
    square_kilometer: 1_000_000,
    square_kilometers: 1_000_000,
    km2: 1_000_000,
    square_foot: 0.092903,
    square_feet: 0.092903,
    ft2: 0.092903,
    acre: 4_046.86,
    acres: 4_046.86,
    hectare: 10_000,
    hectares: 10_000,
  },
  speed: {
    meter_per_second: 1,
    meters_per_second: 1,
    mps: 1,
    kilometer_per_hour: 0.277778,
    kilometers_per_hour: 0.277778,
    kmh: 0.277778,
    kph: 0.277778,
    mile_per_hour: 0.44704,
    miles_per_hour: 0.44704,
    mph: 0.44704,
    knot: 0.514444,
    knots: 0.514444,
    kt: 0.514444,
  },
}

function findCategory(source: string, target: string): string | undefined {
  return Object.entries(UNIT_CONVERSIONS).find(([, conversions]) => (
    conversions[source] !== undefined && conversions[target] !== undefined
  ))?.[0]
}

function convertTemperature(value: number, source: string, target: string): number {
  const celsius = source === 'fahrenheit' || source === 'f'
    ? (value - 32) * 5 / 9
    : source === 'kelvin' || source === 'k'
      ? value - 273.15
      : value
  if (target === 'fahrenheit' || target === 'f') {
    return celsius * 9 / 5 + 32
  }
  return target === 'kelvin' || target === 'k' ? celsius + 273.15 : celsius
}
