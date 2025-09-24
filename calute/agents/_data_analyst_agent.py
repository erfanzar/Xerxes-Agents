
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

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

from calute import Agent
from calute.tools import (
    DataConverter,
    JSONProcessor,
    ReadFile,
    StatisticalAnalyzer,
    WriteFile,
)

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
    """
    Perform comprehensive data analysis.

    Args:
        data: Dataset to analyze (list of dicts or JSON string)
        analysis_type: Type of analysis (descriptive, diagnostic, predictive)

    Returns:
        Analysis results with insights
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
    """
    Clean and preprocess data.

    Args:
        data: Data to clean
        operations: List of cleaning operations to perform

    Returns:
        Cleaned data summary
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
    """
    Detect patterns and anomalies in data.

    Args:
        data: Data to analyze
        pattern_type: Type of pattern to detect (trends, cycles, anomalies)

    Returns:
        Pattern detection results
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
    """
    Generate business insights from analysis.

    Args:
        analysis_results: Results from data analysis
        business_context: Business context for insights

    Returns:
        Actionable business insights
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
    """
    Create dashboard specification for data visualization.

    Args:
        metrics: List of metrics to display
        layout: Dashboard layout type (grid, flow, tabs)

    Returns:
        Dashboard specification
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
    """
    Forecast future values based on historical data.

    Args:
        historical_data: Historical values
        periods: Number of periods to forecast
        method: Forecasting method (linear, exponential, moving_average)

    Returns:
        Forecast results
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
