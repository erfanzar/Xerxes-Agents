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

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .types import CortexAgent, CortexTask, CortexTaskOutput


@dataclass
class PerformanceMetrics:
    """Performance metrics for agents and tasks"""

    execution_time: float
    token_usage: int
    success: bool
    quality_score: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and analyze cortex performance"""

    def __init__(self):
        self.metrics: dict[str, list[PerformanceMetrics]] = {}
        self.alerts: list[dict[str, Any]] = []
        self.chain_metrics: dict[str, list[dict[str, Any]]] = {}
        self.thresholds = {
            "execution_time": 60.0,  # seconds
            "token_usage": 1000,
            "quality_score": 0.7,
            "success_rate": 0.8,
        }

    def record_task_execution(self, task_id: str, agent_id: str, metrics: PerformanceMetrics):
        """Record task execution metrics"""
        key = f"{agent_id}:{task_id}"
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append(metrics)

        # Check for alerts
        self._check_alerts(key, metrics)

    def record_chain_execution(self, chain_name: str, result: dict[str, Any]):
        """Record chain execution metrics"""
        if chain_name not in self.chain_metrics:
            self.chain_metrics[chain_name] = []

        chain_metric = {
            "timestamp": datetime.now(),
            "chain_name": chain_name,
            "chain_type": result.get("chain_type", "unknown"),
            "execution_time": result.get("execution_time", 0),
            "steps_completed": len(result.get("intermediate_results", [])),
            "final_output_size": len(str(result.get("final_output", ""))),
            "success": not result.get("error", False),
        }

        self.chain_metrics[chain_name].append(chain_metric)

        # Check if chain execution was slow
        if chain_metric["execution_time"] > self.thresholds["execution_time"]:
            self.alerts.append(
                {
                    "type": "slow_chain_execution",
                    "chain_name": chain_name,
                    "execution_time": chain_metric["execution_time"],
                    "timestamp": datetime.now(),
                }
            )

    def _check_alerts(self, key: str, metrics: PerformanceMetrics):
        """Check if metrics trigger any alerts"""
        if metrics.execution_time > self.thresholds["execution_time"]:
            self.alerts.append(
                {
                    "type": "slow_execution",
                    "key": key,
                    "value": metrics.execution_time,
                    "threshold": self.thresholds["execution_time"],
                    "timestamp": datetime.now(),
                }
            )

        if metrics.quality_score < self.thresholds["quality_score"]:
            self.alerts.append(
                {
                    "type": "low_quality",
                    "key": key,
                    "value": metrics.quality_score,
                    "threshold": self.thresholds["quality_score"],
                    "timestamp": datetime.now(),
                }
            )

    def get_agent_performance_summary(self, agent_id: str, time_window: timedelta | None = None) -> dict[str, Any]:
        """Get performance summary for an agent"""
        agent_metrics = []
        cutoff_time = datetime.now() - time_window if time_window else None

        for key, metrics_list in self.metrics.items():
            if key.startswith(f"{agent_id}:"):
                for metric in metrics_list:
                    if not cutoff_time or metric.timestamp > cutoff_time:
                        agent_metrics.append(metric)

        if not agent_metrics:
            return {"error": "No metrics found for agent"}

        return {
            "agent_id": agent_id,
            "total_executions": len(agent_metrics),
            "success_rate": sum(1 for m in agent_metrics if m.success) / len(agent_metrics),
            "average_execution_time": statistics.mean(m.execution_time for m in agent_metrics),
            "average_quality_score": statistics.mean(m.quality_score for m in agent_metrics),
            "total_tokens_used": sum(m.token_usage for m in agent_metrics),
            "recent_alerts": [a for a in self.alerts if agent_id in a.get("key", "")],
        }

    def identify_bottlenecks(self) -> list[dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Analyze execution times
        for key, metrics_list in self.metrics.items():
            if len(metrics_list) >= 3:  # Need enough data
                recent_metrics = metrics_list[-10:]  # Last 10 executions
                avg_time = statistics.mean(m.execution_time for m in recent_metrics)

                if avg_time > self.thresholds["execution_time"]:
                    bottlenecks.append(
                        {
                            "type": "slow_execution",
                            "key": key,
                            "average_time": avg_time,
                            "recent_executions": len(recent_metrics),
                            "recommendation": "Consider optimizing task or switching to a faster agent",
                        }
                    )

        for key, metrics_list in self.metrics.items():
            if len(metrics_list) >= 5:
                recent_metrics = metrics_list[-10:]
                success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)

                if success_rate < self.thresholds["success_rate"]:
                    bottlenecks.append(
                        {
                            "type": "low_success_rate",
                            "key": key,
                            "success_rate": success_rate,
                            "recommendation": "Review task requirements or provide additional training",
                        }
                    )

        return bottlenecks

    def generate_optimization_report(self) -> dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_executions": sum(len(metrics) for metrics in self.metrics.values()),
            "bottlenecks": self.identify_bottlenecks(),
            "agent_rankings": self._rank_agents(),
            "task_difficulty_analysis": self._analyze_task_difficulty(),
            "recommendations": self._generate_recommendations(),
        }

    def _rank_agents(self) -> list[dict[str, Any]]:
        """Rank agents by performance"""
        agent_scores = {}

        for key, metrics_list in self.metrics.items():
            agent_id = key.split(":")[0]
            if agent_id not in agent_scores:
                agent_scores[agent_id] = []

            # Calculate composite score
            for metric in metrics_list:
                score = (
                    (1.0 if metric.success else 0.0) * 0.4
                    + metric.quality_score * 0.4
                    + (1.0 - min(metric.execution_time / 60, 1.0)) * 0.2
                )
                agent_scores[agent_id].append(score)

        # Calculate average scores
        rankings = []
        for agent_id, scores in agent_scores.items():
            rankings.append({"agent_id": agent_id, "average_score": statistics.mean(scores), "total_tasks": len(scores)})

        return sorted(rankings, key=lambda x: x["average_score"], reverse=True)

    def _analyze_task_difficulty(self) -> dict[str, Any]:
        """Analyze task difficulty based on metrics"""
        task_analysis = {}

        for key, metrics_list in self.metrics.items():
            task_id = key.split(":")[1]
            if task_id not in task_analysis:
                task_analysis[task_id] = {"execution_times": [], "success_rates": [], "quality_scores": []}

            for metric in metrics_list:
                task_analysis[task_id]["execution_times"].append(metric.execution_time)
                task_analysis[task_id]["success_rates"].append(1.0 if metric.success else 0.0)
                task_analysis[task_id]["quality_scores"].append(metric.quality_score)

        # Calculate difficulty scores
        difficulty_scores = {}
        for task_id, data in task_analysis.items():
            difficulty_scores[task_id] = {
                "difficulty_score": (
                    statistics.mean(data["execution_times"]) / 30 * 0.3
                    + (1 - statistics.mean(data["success_rates"])) * 0.4
                    + (1 - statistics.mean(data["quality_scores"])) * 0.3
                ),
                "average_execution_time": statistics.mean(data["execution_times"]),
                "success_rate": statistics.mean(data["success_rates"]),
                "average_quality": statistics.mean(data["quality_scores"]),
            }

        return difficulty_scores

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Check for slow agents
        rankings = self._rank_agents()
        if rankings and rankings[-1]["average_score"] < 0.5:
            recommendations.append(
                f"Consider retraining or replacing agent {rankings[-1]['agent_id']} "
                f"(score: {rankings[-1]['average_score']:.2f})"
            )

        # Check for difficult tasks
        task_difficulty = self._analyze_task_difficulty()
        for task_id, difficulty in task_difficulty.items():
            if difficulty["difficulty_score"] > 0.7:
                recommendations.append(
                    f"CortexTask {task_id} is particularly difficult. Consider "
                    "breaking it down or providing more context."
                )

        # Check token usage
        total_tokens = sum(m.token_usage for metrics in self.metrics.values() for m in metrics)
        if total_tokens > 10000:
            recommendations.append(
                "High token usage detected. Consider optimizing prompts or using more efficient models."
            )

        return recommendations


class LearningEngine:
    """Learning engine for continuous improvement"""

    def __init__(self):
        self.experience_buffer: list[dict[str, Any]] = []
        self.learned_patterns: dict[str, Any] = {}
        self.improvement_suggestions: list[dict[str, Any]] = []

    def record_experience(
        self,
        task: "CortexTask",
        agent: "CortexAgent",
        input_data: Any,
        output: CortexTaskOutput,
        feedback: dict[str, Any] | None = None,
    ):
        """Record an experience for learning"""
        experience = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.id,
            "task_description": task.description,
            "agent_id": agent.id,
            "agent_role": agent.role,
            "input_summary": str(input_data)[:200],
            "output_summary": output.summary,
            "success": output.status == "success",
            "quality_indicators": self._extract_quality_indicators(output),
            "feedback": feedback or {},
        }

        self.experience_buffer.append(experience)

        # Trigger learning if buffer is large enough
        if len(self.experience_buffer) >= 10:
            self._learn_from_experiences()

    def _extract_quality_indicators(self, output: CortexTaskOutput) -> dict[str, Any]:
        """Extract quality indicators from output"""
        return {
            "output_length": len(output.raw),
            "has_summary": bool(output.summary),
            "completion_time": (output.timestamp - datetime.now()).total_seconds()
            if hasattr(output, "start_time")
            else None,
            "contains_errors": "error" in output.raw.lower() or "failed" in output.raw.lower(),
        }

    def _learn_from_experiences(self):
        """Learn patterns from accumulated experiences"""
        # Group experiences by task type
        task_groups = {}
        for exp in self.experience_buffer:
            task_desc = exp["task_description"]
            if task_desc not in task_groups:
                task_groups[task_desc] = []
            task_groups[task_desc].append(exp)

        # Analyze each task group
        for task_desc, experiences in task_groups.items():
            successful_exps = [e for e in experiences if e["success"]]
            failed_exps = [e for e in experiences if not e["success"]]

            if successful_exps:
                # Learn from successful patterns
                pattern = {
                    "task_description": task_desc,
                    "best_agents": self._identify_best_agents(successful_exps),
                    "success_indicators": self._extract_success_patterns(successful_exps),
                    "average_quality": statistics.mean(e["quality_indicators"]["output_length"] for e in successful_exps)
                    / 1000,  # Normalize
                }

                self.learned_patterns[task_desc] = pattern

            if failed_exps and len(failed_exps) > 2:
                # Generate improvement suggestions
                self.improvement_suggestions.append(
                    {
                        "task_description": task_desc,
                        "failure_rate": len(failed_exps) / len(experiences),
                        "common_issues": self._identify_common_issues(failed_exps),
                        "suggested_improvements": self._generate_improvements(failed_exps),
                    }
                )

        # Clear old experiences
        self.experience_buffer = self.experience_buffer[-50:]  # Keep last 50

    def _identify_best_agents(self, experiences: list[dict[str, Any]]) -> list[str]:
        """Identify best performing agents for a task"""
        agent_performance = {}

        for exp in experiences:
            agent_id = exp["agent_id"]
            if agent_id not in agent_performance:
                agent_performance[agent_id] = {"count": 0, "quality_sum": 0}

            agent_performance[agent_id]["count"] += 1
            agent_performance[agent_id]["quality_sum"] += exp["quality_indicators"]["output_length"] / 1000

        # Calculate average quality per agent
        agent_scores = []
        for agent_id, perf in agent_performance.items():
            avg_quality = perf["quality_sum"] / perf["count"]
            agent_scores.append((agent_id, avg_quality))

        # Return top agents
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in agent_scores[:3]]

    def _extract_success_patterns(self, experiences: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract patterns from successful experiences"""
        patterns = {
            "average_output_length": statistics.mean(e["quality_indicators"]["output_length"] for e in experiences),
            "common_keywords": self._extract_common_keywords(experiences),
            "optimal_conditions": {
                "has_summary": sum(1 for e in experiences if e["quality_indicators"]["has_summary"]) / len(experiences)
            },
        }

        return patterns

    def _extract_common_keywords(self, experiences: list[dict[str, Any]]) -> list[str]:
        """Extract common keywords from successful outputs"""
        # Simple keyword extraction (in practice, use NLP)
        word_counts = {}

        for exp in experiences:
            words = exp["output_summary"].lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]

    def _identify_common_issues(self, failed_experiences: list[dict[str, Any]]) -> list[str]:
        """Identify common issues in failed experiences"""
        issues = []

        # Check for common error indicators
        error_keywords = ["error", "failed", "unable", "cannot", "invalid"]
        keyword_counts = {kw: 0 for kw in error_keywords}

        for exp in failed_experiences:
            output_lower = exp["output_summary"].lower()
            for keyword in error_keywords:
                if keyword in output_lower:
                    keyword_counts[keyword] += 1

        # Report frequent issues
        for keyword, count in keyword_counts.items():
            if count > len(failed_experiences) * 0.3:  # 30% threshold
                issues.append(f"Frequent '{keyword}' errors")

        return issues

    def _generate_improvements(self, failed_experiences: list[dict[str, Any]]) -> list[str]:
        """Generate improvement suggestions based on failures"""
        suggestions = []

        # Analyze quality indicators
        avg_length = statistics.mean(e["quality_indicators"]["output_length"] for e in failed_experiences)

        if avg_length < 100:
            suggestions.append("Outputs are too short. Consider providing more detailed instructions.")

        # Check for specific agents failing frequently
        agent_failures = {}
        for exp in failed_experiences:
            agent_id = exp["agent_id"]
            agent_failures[agent_id] = agent_failures.get(agent_id, 0) + 1

        for agent_id, failure_count in agent_failures.items():
            if failure_count > len(failed_experiences) * 0.5:
                suggestions.append(
                    f"Agent {agent_id} has high failure rate. Consider additional training or using a different agent."
                )

        return suggestions

    def get_recommendations_for_task(self, task_description: str) -> dict[str, Any]:
        """Get learned recommendations for a task"""
        # Find similar learned patterns
        best_match = None
        best_similarity = 0

        for learned_task, pattern in self.learned_patterns.items():
            # Simple similarity check (in practice, use embeddings)
            similarity = len(set(task_description.lower().split()) & set(learned_task.lower().split())) / len(
                set(task_description.lower().split())
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern

        if best_match and best_similarity > 0.3:
            return {
                "recommended_agents": best_match["best_agents"],
                "expected_quality": best_match["average_quality"],
                "success_indicators": best_match["success_indicators"],
                "confidence": best_similarity,
            }

        return {
            "message": "No learned patterns found for this task type",
            "recommendation": "Proceed with default configuration",
        }
