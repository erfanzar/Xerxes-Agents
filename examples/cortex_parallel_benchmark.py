#!/usr/bin/env python3
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
"""Parallel Cortex benchmark for measuring inference engine throughput.

This script is intentionally synthetic. It avoids tools and deep-search logic so
the measurement is dominated by model prefill + decode under concurrent load.

Example:
    ./.venv/bin/python examples/cortex_parallel_benchmark.py -n 64 --min-context-tokens 14000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xerxes import Cortex, CortexAgent, CortexTask, ProcessType, create_llm  # noqa: E402
from xerxes.context.token_counter import TIKTOKEN_AVAILABLE, ProviderTokenCounter  # noqa: E402

ENV_PY_LLM_CONFIG = {
    "provider": "openai",
    "base_url": "http://XXX:11556/v1/",
    "api_key": "sk-xxx",
}
DEFAULT_MODEL = "qwen3-8.19b"
DEFAULT_AGENTS = 64
DEFAULT_MIN_CONTEXT_TOKENS = 14000
DEFAULT_CONTEXT_BUFFER_TOKENS = 512
DEFAULT_MAX_OUTPUT_TOKENS = 64


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""
    value = value.strip().lower().replace(" ", "-")
    return "".join(char for char in value if char.isalnum() or char == "-").strip("-") or "benchmark"


def count_tokens(text: str, model: str | None = None) -> int:
    """Approximate token counting using OpenAI-compatible encoding."""
    return ProviderTokenCounter.count_tokens_for_provider(
        text,
        provider="openai",
        model=model,
    )


def percentile(values: list[float], q: float) -> float:
    """Simple percentile helper."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def build_llm_from_env_config():
    """Create an LLM client using the same settings as env.py."""
    return create_llm(
        ENV_PY_LLM_CONFIG["provider"],
        base_url=ENV_PY_LLM_CONFIG["base_url"],
        api_key=ENV_PY_LLM_CONFIG["api_key"],
    )


def build_output_dir(label: str, base_dir: str | Path = "outputs/cortex_parallel_benchmark") -> Path:
    """Create a timestamped output directory for a benchmark run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{stamp}_{slugify(label)[:60]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_sentinel(label: str, worker_index: int) -> str:
    """Build a deterministic validation sentinel."""
    digest = hashlib.sha1(f"{label}:{worker_index}".encode()).hexdigest()[:16]
    return f"SENTINEL-{worker_index:03d}-{digest}"


def build_context_block(label: str, worker_index: int, block_index: int) -> str:
    """Build one synthetic context block.

    The content is deliberately verbose and unique enough per worker/block to
    reduce the chance of prefix-caching turning this into a misleading test.
    """
    trace_terms = " ".join([f"trace_{worker_index}_{block_index}_{offset}" for offset in range(24)])
    return f"""
Benchmark label: {label}
Worker id: {worker_index}
Block id: {block_index}

This is synthetic benchmark context designed to occupy prompt capacity and stress
the inference engine under concurrent prefill load. The text is intentionally
structured, repetitive, and long enough to create measurable prompt processing
work while still being easy for the model to acknowledge with a tiny response.

Operational notes:
- This worker should treat the context as a dossier, not as instructions to browse.
- The benchmark is measuring inference throughput, not tool latency.
- The context includes repeated but slightly varied language to avoid exact prompt duplication.
- The final validation sentinel appears near the end of the full prompt context.

Technical profile:
The system is simulating request fan-out across many concurrent agents. Each agent
receives a large context payload, a compact instruction set, and a requirement to
emit a short deterministic acknowledgement. This emphasizes prompt ingestion and
backend scheduling behavior more than answer quality. The benchmark intentionally
avoids retrieval tools, file reads, network fetches, or external state that would
pollute latency measurements.

Trace terms:
{trace_terms}

Narrative:
The benchmark dossier for worker {worker_index} and block {block_index} describes
synthetic telemetry, scheduler pressure, queue depth, prefill utilization, and
decode overlap. It references batch construction, KV cache pressure, prompt
serialization, token budgeting, head-of-line blocking, and request completion
variance. These phrases are present only to enlarge and diversify the prompt so
that the engine must process a meaningful amount of text before emitting the final
acknowledgement token sequence.
""".strip()


def build_large_context(
    label: str,
    worker_index: int,
    min_context_tokens: int,
    context_buffer_tokens: int,
    model: str,
) -> tuple[str, int, str]:
    """Build a context that meets or exceeds the requested token budget."""
    sentinel = build_sentinel(label=label, worker_index=worker_index)
    target_tokens = max(1, min_context_tokens + context_buffer_tokens)

    preamble = f"""
Parallel inference benchmark context.
Worker {worker_index} is part of a synchronized load wave.
The model should return only the validation sentinel located at the end.
""".strip()
    footer = f"""
Validation rule:
Return the following sentinel exactly and nothing else.
{sentinel}
""".strip()

    sample_block = build_context_block(label=label, worker_index=worker_index, block_index=0)
    sample_block_tokens = max(1, count_tokens(sample_block, model=model))
    fixed_tokens = count_tokens(f"{preamble}\n\n{footer}", model=model)
    blocks_needed = max(1, math.ceil(max(0, target_tokens - fixed_tokens) / sample_block_tokens))

    parts = [preamble]
    for block_index in range(blocks_needed):
        parts.append(build_context_block(label=label, worker_index=worker_index, block_index=block_index))
    parts.append(footer)

    context = "\n\n".join(parts)
    context_tokens = count_tokens(context, model=model)

    next_block_index = blocks_needed
    while context_tokens < target_tokens:
        parts.insert(-1, build_context_block(label=label, worker_index=worker_index, block_index=next_block_index))
        next_block_index += 1
        context = "\n\n".join(parts)
        context_tokens = count_tokens(context, model=model)

    return context, context_tokens, sentinel


def create_benchmark_agent(llm, model: str, worker_index: int, max_output_tokens: int, verbose: bool) -> CortexAgent:
    """Create one synthetic benchmark worker."""
    return CortexAgent(
        role=f"Load Test Worker {worker_index}",
        goal="Consume a large prompt context and return a tiny deterministic acknowledgement.",
        backstory=(
            "You are a benchmark worker used to stress a parallel inference backend. "
            "You do not browse, plan, or use tools. You only read the provided prompt and return the requested token."
        ),
        instructions=(
            "You are participating in an inference benchmark.\n"
            "Return only the exact validation sentinel requested by the task.\n"
            "Do not explain, summarize, or add punctuation.\n"
            "No markdown, no code fences, no extra text."
        ),
        model=model,
        llm=llm,
        temperature=0.0,
        max_tokens=max_output_tokens,
        verbose=verbose,
        memory_enabled=False,
    )


def estimate_request_tokens(agent: CortexAgent, task: CortexTask, model: str) -> dict[str, int]:
    """Estimate request token usage for one task."""
    task_prompt = f"{task.description}\n\nExpected Output: {task.expected_output}"
    if task.prompt_context:
        task_prompt += f"\n\nAdditional Context: {task.prompt_context}"

    rendered_task_prompt = agent._template_engine.render_task_prompt(
        description=task_prompt,
        expected_output="",
        context="",
    )

    system_tokens = count_tokens(agent.instructions or "", model=model)
    prompt_tokens = count_tokens(rendered_task_prompt, model=model)
    return {
        "system_tokens": system_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": system_tokens + prompt_tokens,
    }


def build_benchmark_cortex(
    label: str,
    model: str,
    agents_count: int,
    min_context_tokens: int,
    context_buffer_tokens: int,
    max_output_tokens: int,
    verbose: bool,
) -> tuple[Cortex, dict[str, Any]]:
    """Build a parallel Cortex benchmark with one task per agent."""
    llm = build_llm_from_env_config()
    output_dir = build_output_dir(label=label)

    agents = []
    tasks = []
    worker_specs = []

    for worker_index in range(1, agents_count + 1):
        agent = create_benchmark_agent(
            llm=llm,
            model=model,
            worker_index=worker_index,
            max_output_tokens=max_output_tokens,
            verbose=verbose,
        )
        context, context_tokens, sentinel = build_large_context(
            label=label,
            worker_index=worker_index,
            min_context_tokens=min_context_tokens,
            context_buffer_tokens=context_buffer_tokens,
            model=model,
        )
        task = CortexTask(
            description=(
                f"Worker {worker_index}: read the entire benchmark context and return the validation sentinel exactly.\n"
                "Your response must contain only the sentinel string."
            ),
            expected_output="The exact validation sentinel string and nothing else.",
            agent=agent,
            prompt_context=context,
            save_to_memory=False,
            max_retries=1,
        )
        token_estimate = estimate_request_tokens(agent=agent, task=task, model=model)

        agents.append(agent)
        tasks.append(task)
        worker_specs.append(
            {
                "worker_index": worker_index,
                "sentinel": sentinel,
                "prompt_context_tokens": context_tokens,
                "estimated_request_tokens": token_estimate["total_tokens"],
                "estimated_system_tokens": token_estimate["system_tokens"],
                "estimated_prompt_tokens": token_estimate["prompt_tokens"],
            }
        )

    cortex = Cortex(
        agents=agents,
        tasks=tasks,
        llm=llm,
        process=ProcessType.PARALLEL,
        verbose=verbose,
        model=model,
        cortex_name=f"Parallel Benchmark x{agents_count}",
        parallel_max_workers=agents_count,
        memory_config={
            "enable_short_term": False,
            "enable_long_term": False,
            "enable_entity": False,
            "enable_user": False,
        },
    )

    benchmark_spec = {
        "label": label,
        "model": model,
        "agents_count": agents_count,
        "min_context_tokens": min_context_tokens,
        "context_buffer_tokens": context_buffer_tokens,
        "max_output_tokens": max_output_tokens,
        "token_counter_backend": "tiktoken" if TIKTOKEN_AVAILABLE else "char_fallback",
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "responses_path": str(output_dir / "responses.json"),
        "worker_specs": worker_specs,
        "llm_config": ENV_PY_LLM_CONFIG,
        "llm_model_info": llm.get_model_info(),
    }
    return cortex, benchmark_spec


def run_benchmark(
    label: str,
    model: str,
    agents_count: int,
    min_context_tokens: int,
    context_buffer_tokens: int,
    max_output_tokens: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the benchmark and return summary + responses payload."""
    cortex, benchmark_spec = build_benchmark_cortex(
        label=label,
        model=model,
        agents_count=agents_count,
        min_context_tokens=min_context_tokens,
        context_buffer_tokens=context_buffer_tokens,
        max_output_tokens=max_output_tokens,
        verbose=verbose,
    )

    started_at = datetime.now().isoformat()
    wall_start = time.perf_counter()
    result = cortex.kickoff(use_streaming=False)
    wall_time = time.perf_counter() - wall_start

    worker_specs = benchmark_spec["worker_specs"]
    task_outputs = result.task_outputs
    task_times = [task_output.execution_time for task_output in task_outputs]

    responses = []
    validated = 0
    for worker_spec, task_output in zip(worker_specs, task_outputs, strict=False):
        response_text = task_output.output.strip()
        response_ok = worker_spec["sentinel"] in response_text
        if response_ok:
            validated += 1
        responses.append(
            {
                "worker_index": worker_spec["worker_index"],
                "expected_sentinel": worker_spec["sentinel"],
                "response": response_text,
                "contains_expected_sentinel": response_ok,
                "execution_time_seconds": task_output.execution_time,
                "estimated_request_tokens": worker_spec["estimated_request_tokens"],
                "prompt_context_tokens": worker_spec["prompt_context_tokens"],
            }
        )

    estimated_total_request_tokens = sum(spec["estimated_request_tokens"] for spec in worker_specs)
    prompt_context_token_counts = [spec["prompt_context_tokens"] for spec in worker_specs]

    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "label": label,
        "model": model,
        "agents_count": agents_count,
        "parallel_max_workers": agents_count,
        "min_context_tokens": min_context_tokens,
        "context_buffer_tokens": context_buffer_tokens,
        "max_output_tokens": max_output_tokens,
        "token_counter_backend": benchmark_spec["token_counter_backend"],
        "llm_config": benchmark_spec["llm_config"],
        "llm_model_info": benchmark_spec["llm_model_info"],
        "wall_time_seconds": wall_time,
        "cortex_execution_time_seconds": result.execution_time,
        "estimated_total_request_tokens": estimated_total_request_tokens,
        "estimated_total_request_tokens_per_second": (
            estimated_total_request_tokens / wall_time if wall_time > 0 else 0.0
        ),
        "requests_per_second": agents_count / wall_time if wall_time > 0 else 0.0,
        "validated_responses": validated,
        "response_validation_rate": validated / len(responses) if responses else 0.0,
        "prompt_context_tokens": {
            "min": min(prompt_context_token_counts) if prompt_context_token_counts else 0,
            "avg": statistics.mean(prompt_context_token_counts) if prompt_context_token_counts else 0.0,
            "max": max(prompt_context_token_counts) if prompt_context_token_counts else 0,
        },
        "task_execution_time_seconds": {
            "min": min(task_times) if task_times else 0.0,
            "avg": statistics.mean(task_times) if task_times else 0.0,
            "p50": percentile(task_times, 0.50),
            "p95": percentile(task_times, 0.95),
            "max": max(task_times) if task_times else 0.0,
        },
        "artifacts": {
            "output_dir": benchmark_spec["output_dir"],
            "summary_path": benchmark_spec["summary_path"],
            "responses_path": benchmark_spec["responses_path"],
        },
    }

    summary_path = Path(benchmark_spec["summary_path"])
    responses_path = Path(benchmark_spec["responses_path"])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    responses_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")

    return summary, {
        "summary_path": str(summary_path),
        "responses_path": str(responses_path),
        "output_dir": benchmark_spec["output_dir"],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a parallel Cortex inference benchmark.")
    parser.add_argument(
        "label",
        nargs="?",
        default="inference-engine-loadtest",
        help="Label for the benchmark run.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "-n",
        "--agents",
        type=int,
        default=DEFAULT_AGENTS,
        help=f"Number of agents/tasks to launch in parallel. Default: {DEFAULT_AGENTS}",
    )
    parser.add_argument(
        "--min-context-tokens",
        type=int,
        default=DEFAULT_MIN_CONTEXT_TOKENS,
        help=f"Minimum prompt-context tokens per agent. Default: {DEFAULT_MIN_CONTEXT_TOKENS}",
    )
    parser.add_argument(
        "--context-buffer-tokens",
        type=int,
        default=DEFAULT_CONTEXT_BUFFER_TOKENS,
        help=f"Extra tokens added above the minimum target for safety. Default: {DEFAULT_CONTEXT_BUFFER_TOKENS}",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Maximum output tokens per agent. Default: {DEFAULT_MAX_OUTPUT_TOKENS}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Cortex/agent logging.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    summary, artifacts = run_benchmark(
        label=args.label,
        model=args.model,
        agents_count=args.agents,
        min_context_tokens=args.min_context_tokens,
        context_buffer_tokens=args.context_buffer_tokens,
        max_output_tokens=args.max_output_tokens,
        verbose=args.verbose,
    )

    print("Parallel Cortex benchmark completed.")
    print(f"Agents: {summary['agents_count']}")
    print(
        f"Prompt context tokens per agent (min/avg/max): {summary['prompt_context_tokens']['min']}/"
        f"{summary['prompt_context_tokens']['avg']:.1f}/{summary['prompt_context_tokens']['max']}"
    )
    print(f"Wall time: {summary['wall_time_seconds']:.2f}s")
    print(f"Estimated request tokens/sec: {summary['estimated_total_request_tokens_per_second']:.2f}")
    print(f"Validated responses: {summary['validated_responses']}/{summary['agents_count']}")
    print(f"Summary: {artifacts['summary_path']}")
    print(f"Responses: {artifacts['responses_path']}")


if __name__ == "__main__":
    main()
