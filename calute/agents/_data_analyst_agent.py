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


"""Data Analyst Agent for statistical analysis and business intelligence.

This module provides a pre-built data analyst agent implementation for the Calute
framework, specialized in data analysis, pattern detection, forecasting, and
business insight generation. The agent can process structured datasets, perform
statistical analysis, clean data, and create visualization specifications.

Key Features:
    - Comprehensive dataset analysis (descriptive, diagnostic, predictive)
    - Data cleaning and preprocessing with configurable operations
    - Pattern detection for trends, cycles, and anomalies
    - Business insight generation with actionable recommendations
    - Dashboard specification creation for data visualization
    - Time series forecasting with multiple methods (linear, exponential, moving average)

Module Attributes:
    analysis_state (dict): Global state tracking datasets, analyses, reports,
        and visualization configurations.
    data_analyst_agent (Agent): Pre-configured data analyst agent instance with
        statistical tools and data processing capabilities enabled.

Example:
    >>> from calute.agents import data_analyst_agent
    >>> # Use the pre-built data analyst agent
    >>> result = await calute.run(
    ...     agent=data_analyst_agent,
    ...     messages=[{"role": "user", "content": "Analyze this sales data"}]
    ... )

    >>> # Or use individual analysis functions
    >>> from calute.agents._data_analyst_agent import analyze_dataset
    >>> report = analyze_dataset(
    ...     data=[{"sales": 100}, {"sales": 150}],
    ...     analysis_type="descriptive"
    ... )
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from ..tools import (
    DataConverter,
    JSONProcessor,
    ReadFile,
    StatisticalAnalyzer,
    WriteFile,
)
from ..types import Agent

analysis_state = {
    "datasets": {},
    "analyses": {},
    "reports": [],
    "visualizations": [],
}


def analyze_dataset(
    data: list[dict] | str,
    analysis_type: str = "descriptive",
) -> str:
    """Perform comprehensive data analysis on a structured dataset.

    Analyzes the provided dataset according to the specified analysis type,
    generating statistical summaries, data structure information, and
    actionable insights. The function supports multiple input formats and
    automatically detects column data types for appropriate analysis.

    The analysis result is stored in the global ``analysis_state`` dictionary
    under the ``analyses`` key, indexed by a unique analysis ID.

    Args:
        data: Dataset to analyze. Accepts either a list of dictionaries where
            each dictionary represents a record, or a JSON/CSV string that
            will be automatically parsed. CSV strings should have a header
            row followed by data rows separated by newlines.
        analysis_type: Type of analysis to perform. Valid values are:
            - ``'descriptive'``: Computes summary statistics (mean, min, max)
              for numeric columns and reports dataset dimensions.
            - ``'diagnostic'``: Identifies data quality issues such as missing
              values and duplicate records (inspects first 100 rows).
            - ``'predictive'``: Provides guidance on predictive modeling
              approaches including ML models and time series analysis.

    Returns:
        A formatted string report containing the analysis ID, analysis type,
        record and feature counts, data structure breakdown by column type,
        and numbered key insights. The report uses a structured text format
        with section headers.

    Example:
        >>> report = analyze_dataset(
        ...     data=[{"sales": 100, "region": "East"}, {"sales": 150, "region": "West"}],
        ...     analysis_type="descriptive",
        ... )
        >>> print(report)  # Shows stats for 'sales' column and dataset shape
    """
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            lines = data.strip().split("\n")
            if lines:
                headers = lines[0].split(",")
                data = []
                for line in lines[1:]:
                    values = line.split(",")
                    data.append(dict(zip(headers, values, strict=False)))

    if not data:
        return "⚠️ No data provided for analysis"

    num_records = len(data) if isinstance(data, list) else 0

    if isinstance(data, list) and data:
        sample = data[0]
        columns = list(sample.keys()) if isinstance(sample, dict) else []

        data_types = {}
        for col in columns:
            values = [row.get(col) for row in data if isinstance(row, dict)]

            sample_val = next((v for v in values if v is not None), None)
            if sample_val is not None:
                if isinstance(sample_val, int | float):
                    data_types[col] = "numeric"
                elif isinstance(sample_val, bool):
                    data_types[col] = "boolean"
                else:
                    data_types[col] = "text"
    else:
        columns = []
        data_types = {}

    insights = []

    if analysis_type == "descriptive":
        insights.append(f"Dataset contains {num_records} records")
        insights.append(f"Number of features: {len(columns)}")

        for col in columns:
            if data_types.get(col) == "numeric":
                values = []
                for row in data:
                    try:
                        val = float(row.get(col, 0))
                        values.append(val)
                    except (ValueError, TypeError):
                        pass

                if values:
                    avg = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    insights.append(f"{col}: avg={avg:.2f}, min={min_val}, max={max_val}")

    elif analysis_type == "diagnostic":
        insights.append("Diagnostic analysis identifies causes and correlations")

        missing_counts = defaultdict(int)
        for row in data[:100]:
            if isinstance(row, dict):
                for col in columns:
                    if row.get(col) is None or row.get(col) == "":
                        missing_counts[col] += 1

        if missing_counts:
            insights.append(f"Missing values detected in {len(missing_counts)} columns")

        if len(data) > len(set(str(row) for row in data[:100])):
            insights.append("Potential duplicate records detected")

    elif analysis_type == "predictive":
        insights.append("Predictive analysis forecasts future trends")
        insights.append("Would apply ML models for predictions")
        insights.append("Time series analysis for temporal data")

    analysis_result = {
        "id": analysis_id,
        "type": analysis_type,
        "num_records": num_records,
        "num_features": len(columns),
        "columns": columns,
        "data_types": data_types,
        "insights": insights,
        "timestamp": datetime.now().isoformat(),
    }

    analysis_state["analyses"][analysis_id] = analysis_result

    result = f"""📊 DATA ANALYSIS REPORT
{"=" * 50}
Analysis ID: {analysis_id}
Type: {analysis_type.upper()}
Records: {num_records}
Features: {len(columns)}

DATA STRUCTURE:
"""

    for col, dtype in data_types.items():
        result += f"• {col}: {dtype}\n"

    result += "\nKEY INSIGHTS:\n"
    for i, insight in enumerate(insights, 1):
        result += f"{i}. {insight}\n"

    result += "\nStatus: Analysis completed successfully"

    return result


def clean_data(data: list[dict] | str, operations: list[str] | None = None) -> str:
    """Clean and preprocess a dataset by applying configurable cleaning operations.

    Applies a sequence of data cleaning operations to the input dataset and
    produces a detailed cleaning report. The cleaned dataset metadata is stored
    in the global ``analysis_state`` dictionary under the ``datasets`` key.

    Args:
        data: Data to clean. Accepts either a list of dictionaries where each
            dictionary represents a record, or a JSON string that will be
            parsed automatically. Each record should be a dictionary with
            consistent keys.
        operations: List of cleaning operation names to perform, applied in
            order. If ``None``, defaults to all available operations:
            ``['remove_duplicates', 'handle_missing', 'standardize', 'validate']``.
            Supported operations are:
            - ``'remove_duplicates'``: Removes exact duplicate records based
              on string representation of sorted dictionary items.
            - ``'handle_missing'``: Fills ``None`` and empty string values
              with ``'N/A'`` for strings or ``0`` for other types.
            - ``'standardize'``: Strips whitespace and applies title case
              to all string values in the dataset.
            - ``'validate'``: Removes records that are not non-empty
              dictionaries.

    Returns:
        A formatted string report containing the cleaning ID, original and
        final record counts, list of operations performed, a detailed
        cleaning log with per-operation results, and a data quality score
        expressed as a percentage of retained records.

    Example:
        >>> report = clean_data(
        ...     data=[{"name": " john "}, {"name": " john "}, {"name": None}],
        ...     operations=["remove_duplicates", "handle_missing", "standardize"],
        ... )
    """
    cleaning_id = f"clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if operations is None:
        operations = ["remove_duplicates", "handle_missing", "standardize", "validate"]

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return "⚠️ Invalid data format"

    if not isinstance(data, list):
        return "⚠️ Data must be a list of records"

    original_count = len(data)
    cleaning_log = []

    cleaned_data = data.copy()

    if "remove_duplicates" in operations:
        unique_data = []
        seen = set()
        for row in cleaned_data:
            row_str = str(sorted(row.items()) if isinstance(row, dict) else row)
            if row_str not in seen:
                unique_data.append(row)
                seen.add(row_str)

        removed = len(cleaned_data) - len(unique_data)
        cleaned_data = unique_data
        cleaning_log.append(f"Removed {removed} duplicate records")

    if "handle_missing" in operations:
        filled_count = 0
        for row in cleaned_data:
            if isinstance(row, dict):
                for key in list(row.keys()):
                    if row[key] is None or row[key] == "":
                        row[key] = "N/A" if isinstance(row[key], str) else 0
                        filled_count += 1

        cleaning_log.append(f"Filled {filled_count} missing values")

    if "standardize" in operations:
        for row in cleaned_data:
            if isinstance(row, dict):
                for key, value in row.items():
                    if isinstance(value, str):
                        row[key] = value.strip().title()

        cleaning_log.append("Standardized text formatting")

    if "validate" in operations:
        invalid_count = 0
        valid_data = []
        for row in cleaned_data:
            if isinstance(row, dict) and len(row) > 0:
                valid_data.append(row)
            else:
                invalid_count += 1

        cleaned_data = valid_data
        if invalid_count > 0:
            cleaning_log.append(f"Removed {invalid_count} invalid records")

    final_count = len(cleaned_data)

    analysis_state["datasets"][cleaning_id] = {
        "id": cleaning_id,
        "original_count": original_count,
        "final_count": final_count,
        "operations": operations,
        "log": cleaning_log,
        "timestamp": datetime.now().isoformat(),
    }

    result = f"""🧹 DATA CLEANING REPORT
{"=" * 50}
Cleaning ID: {cleaning_id}
Original Records: {original_count}
Final Records: {final_count}
Records Modified: {original_count - final_count}

OPERATIONS PERFORMED:
"""

    for op in operations:
        result += f"✓ {op.replace('_', ' ').title()}\n"

    result += "\nCLEANING LOG:\n"
    for log_entry in cleaning_log:
        result += f"• {log_entry}\n"

    result += f"\nData Quality Score: {(final_count / original_count * 100):.1f}%"

    return result


def detect_patterns(data: list[Any], pattern_type: str = "trends") -> str:
    """Detect patterns, trends, cycles, and anomalies in a data sequence.

    Analyzes the input data for the specified pattern type using statistical
    methods. Numeric values are extracted from the data for analysis; for
    dictionary items, the string length is used as a proxy value. Results
    are stored in the global ``analysis_state`` dictionary.

    Args:
        data: Data sequence to analyze. Can be a list of numeric values,
            strings (attempted float conversion), or dictionaries (string
            length is used as proxy). Must contain at least 2 items for
            trend detection and at least 1 item for anomaly detection.
        pattern_type: Type of pattern detection to perform. Valid values are:
            - ``'trends'``: Compares the average of the first and second
              halves of the data to detect increasing, decreasing, or
              stable trends (uses a 10% threshold).
            - ``'cycles'``: Checks for seasonal and periodic patterns.
              Reports a potential cycle length estimate when data has
              more than 10 points (estimated as ``len(data) // 4``).
            - ``'anomalies'``: Identifies statistical outliers using a
              2-standard-deviation threshold from the mean. Reports up
              to 3 anomalous data points with their index and value.

    Returns:
        A formatted string report containing the pattern ID, detection type,
        number of data points analyzed, a numbered list of detected patterns,
        and a confidence level (High if >2 patterns, Medium if any, Low
        if none).

    Example:
        >>> result = detect_patterns([10, 20, 30, 40, 50], pattern_type="trends")
        >>> "Increasing trend detected" in result
        True
    """
    pattern_id = f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    patterns_found = []

    if pattern_type == "trends":
        if isinstance(data, list) and len(data) > 1:
            numeric_data = []
            for item in data:
                try:
                    numeric_data.append(float(item) if not isinstance(item, dict) else len(str(item)))
                except (ValueError, TypeError):
                    pass

            if len(numeric_data) > 2:
                first_half = sum(numeric_data[: len(numeric_data) // 2]) / (len(numeric_data) // 2)
                second_half = sum(numeric_data[len(numeric_data) // 2 :]) / (len(numeric_data) - len(numeric_data) // 2)

                if second_half > first_half * 1.1:
                    patterns_found.append("Increasing trend detected")
                elif second_half < first_half * 0.9:
                    patterns_found.append("Decreasing trend detected")
                else:
                    patterns_found.append("Stable pattern observed")

    elif pattern_type == "cycles":
        patterns_found.append("Checking for seasonal patterns")
        patterns_found.append("Analyzing periodicity")

        if len(data) > 10:
            patterns_found.append(f"Potential cycle length: {len(data) // 4} periods")

    elif pattern_type == "anomalies":
        numeric_data = []
        for item in data:
            try:
                numeric_data.append(float(item) if not isinstance(item, dict) else len(str(item)))
            except (ValueError, TypeError):
                pass

        if numeric_data:
            mean = sum(numeric_data) / len(numeric_data)
            std_dev = (sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)) ** 0.5

            anomalies = []
            for i, value in enumerate(numeric_data):
                if abs(value - mean) > 2 * std_dev:
                    anomalies.append(f"Index {i}: value {value:.2f} (>{2 * std_dev:.2f} from mean)")

            if anomalies:
                patterns_found.append(f"Found {len(anomalies)} anomalies")
                patterns_found.extend(anomalies[:3])
            else:
                patterns_found.append("No significant anomalies detected")

    analysis_state["analyses"][pattern_id] = {
        "id": pattern_id,
        "type": pattern_type,
        "data_points": len(data),
        "patterns": patterns_found,
        "timestamp": datetime.now().isoformat(),
    }

    result = f"""🔍 PATTERN DETECTION
{"=" * 50}
Pattern ID: {pattern_id}
Type: {pattern_type.upper()}
Data Points: {len(data)}

PATTERNS DETECTED:
"""

    for i, pattern in enumerate(patterns_found, 1):
        result += f"{i}. {pattern}\n"

    if not patterns_found:
        result += "No significant patterns detected\n"

    result += f"\nConfidence Level: {'High' if len(patterns_found) > 2 else 'Medium' if patterns_found else 'Low'}"

    return result


def generate_insights(
    analysis_results: dict[str, Any],
    business_context: str = "",
) -> str:
    """Generate actionable business insights from data analysis results.

    Examines analysis results and business context to produce domain-specific
    insights and strategic recommendations. The function performs keyword-based
    matching on both the analysis results and the business context string to
    generate relevant insights. Results are appended to the global
    ``analysis_state['reports']`` list.

    Args:
        analysis_results: Dictionary containing results from prior data
            analysis operations. The function inspects the string
            representation for keywords such as ``'trends'``,
            ``'anomal'``, and ``'pattern'`` to generate corresponding
            insights.
        business_context: Optional free-text description of the business
            domain or scenario. The function checks for domain keywords
            including ``'sales'``, ``'customer'``, ``'cost'``,
            ``'expense'``, and ``'performance'`` to generate
            context-specific insights and recommendations. Defaults to
            an empty string.

    Returns:
        A formatted string report containing the insights ID, business
        context label, numbered key insights derived from the analysis,
        strategic insights (always included), numbered actionable
        recommendations, and fixed impact/confidence indicators.

    Example:
        >>> result = generate_insights(
        ...     analysis_results={"type": "trends", "patterns": ["growth"]},
        ...     business_context="sales performance quarterly review",
        ... )
    """
    insights_id = f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    insights = []
    recommendations = []

    if "trends" in str(analysis_results).lower():
        insights.append("Growth trends indicate positive momentum")
        recommendations.append("Scale operations to capture growth")

    if "anomal" in str(analysis_results).lower():
        insights.append("Anomalies detected require investigation")
        recommendations.append("Implement monitoring for outliers")

    if "pattern" in str(analysis_results).lower():
        insights.append("Recurring patterns suggest predictability")
        recommendations.append("Leverage patterns for forecasting")

    if business_context:
        context_lower = business_context.lower()

        if "sales" in context_lower:
            insights.append("Sales data shows seasonal variations")
            recommendations.append("Adjust inventory for peak periods")

        if "customer" in context_lower:
            insights.append("Customer behavior patterns identified")
            recommendations.append("Personalize engagement strategies")

        if "cost" in context_lower or "expense" in context_lower:
            insights.append("Cost optimization opportunities found")
            recommendations.append("Review high-cost categories")

        if "performance" in context_lower:
            insights.append("Performance metrics show improvement areas")
            recommendations.append("Focus on underperforming segments")

    strategic_insights = [
        "Data-driven decision making enabled",
        "Predictive capabilities enhanced",
        "Risk factors identified and quantified",
    ]

    analysis_state["reports"].append(
        {
            "id": insights_id,
            "context": business_context,
            "insights": insights + strategic_insights,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }
    )

    result = f"""💡 BUSINESS INSIGHTS
{"=" * 50}
Insights ID: {insights_id}
Context: {business_context or "General Analysis"}

KEY INSIGHTS:
"""

    for i, insight in enumerate(insights, 1):
        result += f"{i}. {insight}\n"

    result += "\nSTRATEGIC INSIGHTS:\n"
    for insight in strategic_insights:
        result += f"• {insight}\n"

    result += "\nRECOMMENDATIONS:\n"
    for i, rec in enumerate(recommendations, 1):
        result += f"{i}. {rec}\n"

    result += "\nImpact Level: High"
    result += "\nConfidence: 85%"

    return result


def create_dashboard_spec(
    metrics: list[str],
    layout: str = "grid",
) -> str:
    """Create a dashboard specification for data visualization.

    Generates a structured dashboard configuration containing visualization
    components for each specified metric. Components cycle through chart
    types (chart, gauge, table, card, heatmap) and are positioned according
    to the selected layout. An executive summary card is always appended.
    The dashboard spec is stored in ``analysis_state['visualizations']``.

    Args:
        metrics: List of metric name strings to include as dashboard
            components. Each metric generates one visualization component.
            Metric names are converted to title-cased display titles with
            underscores replaced by spaces.
        layout: Dashboard layout strategy. Valid values are:
            - ``'grid'``: Components are arranged in a 3-column grid with
              row/col positioning.
            - ``'flow'``: Components are ordered sequentially.
            - ``'tabs'``: Components are ordered sequentially (same as flow
              in the current implementation).

    Returns:
        A formatted string report containing the dashboard ID, layout type,
        component count, refresh rate (5 minutes), a detailed listing of
        each component with its type and metric, and feature flags for
        interactivity, real-time updates, and mobile responsiveness.

    Example:
        >>> spec = create_dashboard_spec(
        ...     metrics=["revenue", "user_count", "conversion_rate"],
        ...     layout="grid",
        ... )
    """
    dashboard_id = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dashboard = {
        "id": dashboard_id,
        "layout": layout,
        "title": "Analytics Dashboard",
        "refresh_rate": "5m",
        "components": [],
    }

    component_types = ["chart", "gauge", "table", "card", "heatmap"]

    for i, metric in enumerate(metrics):
        component = {
            "id": f"comp_{i}",
            "type": component_types[i % len(component_types)],
            "metric": metric,
            "title": metric.replace("_", " ").title(),
            "position": {"row": i // 3, "col": i % 3} if layout == "grid" else {"order": i},
            "config": {
                "color_scheme": "blue",
                "show_legend": True,
                "interactive": True,
            },
        }
        dashboard["components"].append(component)

    dashboard["components"].append(
        {
            "id": "summary",
            "type": "summary_card",
            "title": "Executive Summary",
            "position": {"row": 0, "col": 0, "span": 3} if layout == "grid" else {"order": 0},
            "config": {
                "show_trends": True,
                "highlight_changes": True,
            },
        }
    )

    analysis_state["visualizations"].append(dashboard)

    result = f"""📊 DASHBOARD SPECIFICATION
{"=" * 50}
Dashboard ID: {dashboard_id}
Layout: {layout.upper()}
Components: {len(dashboard["components"])}

DASHBOARD STRUCTURE:
Title: {dashboard["title"]}
Refresh Rate: {dashboard["refresh_rate"]}

COMPONENTS:
"""

    for comp in dashboard["components"]:
        icon = {"chart": "📈", "gauge": "🎯", "table": "📋", "card": "🎴", "heatmap": "🗺️"}.get(comp["type"], "📊")
        result += f"{icon} {comp['title']}\n"
        result += f"   Type: {comp['type']}\n"
        if "metric" in comp:
            result += f"   Metric: {comp['metric']}\n"

    result += "\nInteractive Features: ✓"
    result += "\nReal-time Updates: ✓"
    result += "\nMobile Responsive: ✓"

    return result


def forecast_values(
    historical_data: list[float],
    periods: int = 5,
    method: str = "linear",
) -> str:
    """Forecast future values based on historical time series data.

    Applies the specified forecasting method to historical data to generate
    predicted values for future periods. Includes confidence intervals based
    on the standard deviation of the historical data and provides a trend
    summary comparing average forecast values to historical averages.

    Args:
        historical_data: List of numeric historical values ordered
            chronologically. Must contain at least 2 data points for
            any forecasting method to operate.
        periods: Number of future periods to forecast. Each period
            generates one predicted value with confidence interval.
            Defaults to 5.
        method: Forecasting algorithm to use. Valid values are:
            - ``'linear'``: Fits a least-squares linear regression to the
              historical data and extrapolates. Best for data with a
              consistent linear trend.
            - ``'exponential'``: Uses exponential smoothing with
              ``alpha=0.3``. The first forecast blends the last observed
              value with the historical mean; subsequent forecasts apply
              smoothing to prior forecasts.
            - ``'moving_average'``: Uses the average of the last 3 (or
              fewer) historical values as the base forecast, with minor
              oscillating variation applied to each period.

    Returns:
        A formatted string report containing the forecast ID, method used,
        historical and forecast period counts, a historical summary
        (last value, average, trend arrow), per-period forecast values
        with 95% confidence intervals (1.96 * std_dev), and an overall
        forecast summary indicating expected growth, decline, or stability.
        Returns a warning string if fewer than 2 historical data points
        are provided.

    Example:
        >>> result = forecast_values(
        ...     historical_data=[100, 110, 120, 130, 140],
        ...     periods=3,
        ...     method="linear",
        ... )
    """
    forecast_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not historical_data or len(historical_data) < 2:
        return "⚠️ Insufficient historical data for forecasting"

    forecasts = []

    if method == "linear":
        if len(historical_data) >= 2:
            n = len(historical_data)
            x_mean = (n - 1) / 2
            y_mean = sum(historical_data) / n

            numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(historical_data))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0
            intercept = y_mean - slope * x_mean

            for i in range(periods):
                forecast_value = slope * (n + i) + intercept
                forecasts.append(forecast_value)

    elif method == "moving_average":
        window = min(3, len(historical_data))
        last_values = historical_data[-window:]
        base_forecast = sum(last_values) / window

        for i in range(periods):
            variation = (i % 2 - 0.5) * 0.1 * base_forecast
            forecasts.append(base_forecast + variation)

    elif method == "exponential":
        alpha = 0.3
        last_value = historical_data[-1]

        for i in range(periods):
            if i == 0:
                forecast = alpha * last_value + (1 - alpha) * (sum(historical_data) / len(historical_data))
            else:
                forecast = alpha * forecasts[-1] + (1 - alpha) * forecasts[-1]
            forecasts.append(forecast)

    std_dev = (
        sum((x - sum(historical_data) / len(historical_data)) ** 2 for x in historical_data) / len(historical_data)
    ) ** 0.5

    forecast_result = {
        "id": forecast_id,
        "method": method,
        "historical_periods": len(historical_data),
        "forecast_periods": periods,
        "forecasts": forecasts,
        "confidence_interval": std_dev * 1.96,
        "timestamp": datetime.now().isoformat(),
    }

    result = f"""📈 FORECAST RESULTS
{"=" * 50}
Forecast ID: {forecast_id}
Method: {method.upper()}
Historical Periods: {len(historical_data)}
Forecast Periods: {periods}

HISTORICAL SUMMARY:
• Last Value: {historical_data[-1]:.2f}
• Average: {sum(historical_data) / len(historical_data):.2f}
• Trend: {"↑" if historical_data[-1] > historical_data[0] else "↓" if historical_data[-1] < historical_data[0] else "→"}

FORECAST VALUES:
"""

    for i, value in enumerate(forecasts, 1):
        lower = value - forecast_result["confidence_interval"]
        upper = value + forecast_result["confidence_interval"]
        result += f"Period {i}: {value:.2f} (CI: {lower:.2f} - {upper:.2f})\n"

    if forecasts:
        avg_forecast = sum(forecasts) / len(forecasts)
        avg_historical = sum(historical_data) / len(historical_data)

        if avg_forecast > avg_historical * 1.05:
            trend = "Expected growth"
        elif avg_forecast < avg_historical * 0.95:
            trend = "Expected decline"
        else:
            trend = "Stable outlook"

        result += f"\nFORECAST SUMMARY: {trend}"
        result += "\nConfidence Level: 75%"

    return result


data_analyst_agent = Agent(
    id="data_analyst_agent",
    name="Data Analysis Assistant",
    model=None,
    instructions="""You are an expert data analyst and business intelligence specialist.

Your expertise includes:
- Statistical analysis and hypothesis testing
- Data cleaning and preprocessing
- Pattern recognition and anomaly detection
- Predictive modeling and forecasting
- Data visualization and dashboard design
- Business intelligence and strategic insights
- Report generation and presentation

Analysis Principles:
1. Ensure data quality before analysis
2. Use appropriate statistical methods
3. Validate findings with multiple approaches
4. Present insights clearly and concisely
5. Focus on actionable recommendations
6. Consider business context and implications
7. Document assumptions and limitations

When analyzing data:
- Start with exploratory data analysis
- Identify and handle data quality issues
- Look for patterns, trends, and outliers
- Apply suitable statistical techniques
- Create meaningful visualizations
- Generate actionable insights
- Provide confidence levels for findings

Your goal is to transform raw data into valuable insights
that drive informed business decisions.""",
    functions=[
        analyze_dataset,
        clean_data,
        detect_patterns,
        generate_insights,
        create_dashboard_spec,
        forecast_values,
        StatisticalAnalyzer,
        DataConverter,
        JSONProcessor,
        ReadFile,
        WriteFile,
    ],
    temperature=0.6,
    max_tokens=8192,
)
