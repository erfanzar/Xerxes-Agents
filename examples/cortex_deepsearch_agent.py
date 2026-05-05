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
"""Cortex deep-search runner with configurable parallel researchers.

Pipeline:
1. One strategist creates the search plan.
2. `n` web researchers run in parallel on distinct tracks.
3. One analyst and one writer merge the research into a final report.

The generated artifacts are written to `outputs/cortex_deepsearch/<session>/`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from xerxes import Cortex, CortexAgent, CortexOutput, CortexTask, ProcessType, create_llm  # noqa: E402
from xerxes.logging.console import stream_callback  # noqa: E402
from xerxes.tools import GoogleSearch, WebScraper  # noqa: E402

# Keep the same LLM connection settings that are already used in env.py.
ENV_PY_LLM_CONFIG = {
    "provider": "openai",
    "base_url": "http://XXX:11556/v1/",
    "api_key": "sk-xxx",
}
DEFAULT_MODEL = "qwen3-8.19b"
DEFAULT_PARALLEL_RESEARCHERS = 4

TRACK_LIBRARY = [
    {
        "name": "landscape-overview",
        "focus": "Map the space, define the topic clearly, and identify the main actors, categories, and terminology.",
    },
    {
        "name": "recent-developments",
        "focus": "Focus on recent launches, announcements, upgrades, funding, or major shifts tied to the topic.",
    },
    {
        "name": "primary-sources",
        "focus": (
            "Prioritize official docs, company blogs, source repos, product pages, papers, or first-party statements."
        ),
    },
    {
        "name": "benchmarks-and-data",
        "focus": "Look for quantitative evidence, evaluations, benchmarks, datasets, or measurable adoption signals.",
    },
    {
        "name": "open-source-ecosystem",
        "focus": (
            "Investigate open-source projects, libraries, integrations, and implementation patterns around the topic."
        ),
    },
    {
        "name": "real-world-adoption",
        "focus": "Find case studies, production usage reports, customer stories, or practitioner writeups.",
    },
    {
        "name": "risks-and-limitations",
        "focus": "Surface failure modes, critiques, security issues, tradeoffs, limitations, or contested claims.",
    },
    {
        "name": "future-direction",
        "focus": "Look for roadmaps, open problems, strategic implications, and likely next-step developments.",
    },
    {
        "name": "competitive-landscape",
        "focus": "Compare notable vendors, products, approaches, or competing schools of thought.",
    },
    {
        "name": "regulation-and-governance",
        "focus": "Search for policy, compliance, regulatory, or governance angles where relevant.",
    },
    {
        "name": "academic-research",
        "focus": "Search for papers, technical reports, and research-oriented discussion around the topic.",
    },
    {
        "name": "community-signal",
        "focus": "Look for practitioner commentary, community experiments, or reported implementation experience.",
    },
]


def slugify(value: str) -> str:
    """Create a filesystem-safe slug from a topic string."""
    collapsed = re.sub(r"\s+", "-", value.strip().lower())
    slug = re.sub(r"[^a-z0-9-]+", "", collapsed)
    return slug.strip("-") or "deepsearch-topic"


def truncate_for_prompt(text: str, limit: int) -> str:
    """Trim long context blocks so synthesis prompts stay bounded."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n...[truncated for prompt budget]..."


def serialize_paths(value: Any) -> Any:
    """Convert Path-heavy structures into JSON-safe structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_paths(item) for key, item in value.items()}
    return value


def build_llm_from_env_config():
    """Create an LLM client using the same settings as env.py."""
    return create_llm(
        ENV_PY_LLM_CONFIG["provider"],
        base_url=ENV_PY_LLM_CONFIG["base_url"],
        api_key=ENV_PY_LLM_CONFIG["api_key"],
    )


def build_session_dir(topic: str, base_dir: str | Path = "outputs/cortex_deepsearch") -> Path:
    """Create a timestamped output directory for a topic."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base_dir) / f"{stamp}_{slugify(topic)[:60]}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def build_output_files(topic: str, parallel_researchers: int) -> dict[str, Any]:
    """Build the output layout for a run."""
    session_dir = build_session_dir(topic)
    research_files = [
        session_dir / f"02_research_track_{index:02d}_{slugify(track['name'])}.md"
        for index, track in enumerate(build_research_tracks(parallel_researchers), start=1)
    ]
    return {
        "session": session_dir,
        "plan": session_dir / "01_search_plan.md",
        "research": research_files,
        "analysis": session_dir / "03_analysis_brief.md",
        "report": session_dir / "04_final_report.md",
        "metadata": session_dir / "run_metadata.json",
        "memory": session_dir / "cortex_memory.db",
    }


def build_research_tracks(parallel_researchers: int) -> list[dict[str, str]]:
    """Select or extend research tracks for parallel workers."""
    tracks = []
    for index in range(parallel_researchers):
        template = TRACK_LIBRARY[index % len(TRACK_LIBRARY)]
        cycle = index // len(TRACK_LIBRARY)
        if cycle == 0:
            tracks.append(template)
        else:
            tracks.append(
                {
                    "name": f"{template['name']}-pass-{cycle + 1}",
                    "focus": (
                        template["focus"] + " Search for different sources and a different angle than earlier passes."
                    ),
                }
            )
    return tracks


def create_strategist_agent(llm, model: str) -> CortexAgent:
    """Create the planning agent."""
    return CortexAgent(
        role="Search Strategist",
        goal="Design an exhaustive but efficient research plan before gathering evidence.",
        backstory=(
            "You are an experienced OSINT and technical research planner. "
            "You break ambiguous topics into search tracks, define validation checks, "
            "and sequence the research so later stages can gather stronger evidence."
        ),
        instructions=(
            "You create deep-search plans.\n"
            "Always decompose the topic into distinct angles.\n"
            "Prefer recent, high-signal, primary, and domain-authoritative sources.\n"
            "Flag assumptions, ambiguity, and likely failure modes before search begins."
        ),
        model=model,
        llm=llm,
        temperature=0.1,
        max_tokens=4096,
        verbose=True,
    )


def create_researcher_agent(
    llm, model: str, worker_index: int, total_workers: int, track: dict[str, str]
) -> CortexAgent:
    """Create one parallel researcher."""
    return CortexAgent(
        role=f"Web Researcher {worker_index}",
        goal=(
            f"Collect evidence for research track {worker_index}/{total_workers} "
            f"with explicit links, concise summaries, and confidence notes."
        ),
        backstory=(
            f"You are research worker {worker_index} of {total_workers}. "
            f"Your assigned track is '{track['name']}'. You search broadly, scrape only promising pages, "
            "and separate verified findings from weak signals. You do not invent citations."
        ),
        instructions=(
            "You must use the available web tools to gather evidence.\n"
            "Stay focused on your assigned research track and minimize overlap with other workers.\n"
            "Prefer primary sources when possible.\n"
            "If a source is weak, stale, or derivative, say so explicitly."
        ),
        model=model,
        llm=llm,
        tools=[GoogleSearch, WebScraper],
        temperature=0.2,
        max_tokens=4096,
        verbose=True,
    )


def create_analyst_agent(llm, model: str) -> CortexAgent:
    """Create the synthesis agent."""
    return CortexAgent(
        role="Research Analyst",
        goal="Synthesize evidence into findings, disagreements, risks, and open questions.",
        backstory=(
            "You are a senior analyst who merges evidence into a coherent picture, "
            "tracks contradictions, and identifies what still needs validation."
        ),
        instructions=(
            "Synthesize only from the evidence provided in context.\n"
            "Separate strong findings, weak signals, contradictions, and unresolved gaps.\n"
            "Do not introduce new facts unless they are already present in the context."
        ),
        model=model,
        llm=llm,
        temperature=0.2,
        max_tokens=6144,
        verbose=True,
    )


def create_writer_agent(llm, model: str) -> CortexAgent:
    """Create the final report writer."""
    return CortexAgent(
        role="Technical Report Writer",
        goal="Produce a polished deep-search report that is easy to scan and act on.",
        backstory=(
            "You write concise, technical research reports for engineers and operators. "
            "You preserve nuance, cite sources inline, and keep speculation clearly labeled."
        ),
        instructions=(
            "Write in clean markdown.\n"
            "Use sections, short paragraphs, and bullet lists only where they improve scanability.\n"
            "Preserve source links inline using markdown links whenever possible."
        ),
        model=model,
        llm=llm,
        temperature=0.2,
        max_tokens=6144,
        verbose=True,
    )


def build_planning_cortex(topic: str, llm, model: str, files: dict[str, Any]) -> Cortex:
    """Build the planning stage Cortex."""
    strategist = create_strategist_agent(llm=llm, model=model)
    plan_task = CortexTask(
        description=(
            f"Create a deep-search plan for the topic: '{topic}'.\n"
            "Return:\n"
            "1. the exact research objective\n"
            "2. 5-8 sub-questions that should be answered\n"
            "3. 8-12 concrete search queries\n"
            "4. source prioritization guidance\n"
            "5. likely blind spots, biases, and validation checks\n"
            "Keep the plan compact and operational."
        ),
        expected_output="A markdown research plan with search tracks, queries, and validation checks.",
        agent=strategist,
        output_file=str(files["plan"]),
        max_retries=2,
    )
    return Cortex(
        agents=[strategist],
        tasks=[plan_task],
        llm=llm,
        process=ProcessType.SEQUENTIAL,
        verbose=True,
        model=model,
        cortex_name="DeepSearch Planning",
        memory_config={
            "enable_short_term": True,
            "enable_long_term": True,
            "enable_entity": True,
            "short_term_capacity": 100,
            "long_term_capacity": 1500,
            "persistence_path": str(files["memory"]),
        },
    )


def build_parallel_research_cortex(
    topic: str,
    plan_output: str,
    llm,
    model: str,
    files: dict[str, Any],
    parallel_researchers: int,
) -> Cortex:
    """Build the parallel research stage Cortex."""
    tracks = build_research_tracks(parallel_researchers)
    prompt_plan = truncate_for_prompt(plan_output, 5000)

    agents = []
    tasks = []
    for index, track in enumerate(tracks, start=1):
        agent = create_researcher_agent(
            llm=llm,
            model=model,
            worker_index=index,
            total_workers=parallel_researchers,
            track=track,
        )
        task = CortexTask(
            description=(
                f"Research track {index}/{parallel_researchers} for '{topic}'.\n"
                f"Track name: {track['name']}\n"
                f"Track focus: {track['focus']}\n"
                "Requirements:\n"
                "- use GoogleSearch and WebScraper\n"
                "- gather 3-6 high-signal sources for this track\n"
                "- for each source include title, URL, claim, why it matters, and confidence\n"
                "- end with unresolved questions for this track\n"
                "Keep the brief compact and avoid repeating generic facts."
            ),
            expected_output="A compact markdown research brief with sources and track-specific findings.",
            agent=agent,
            prompt_context=f"Search Plan:\n{prompt_plan}",
            output_file=str(files["research"][index - 1]),
            max_retries=2,
        )
        agents.append(agent)
        tasks.append(task)

    return Cortex(
        agents=agents,
        tasks=tasks,
        llm=llm,
        process=ProcessType.PARALLEL,
        verbose=True,
        model=model,
        cortex_name=f"DeepSearch Research x{parallel_researchers}",
        memory_config={
            "enable_short_term": True,
            "enable_long_term": True,
            "enable_entity": True,
            "short_term_capacity": 100,
            "long_term_capacity": 1500,
            "persistence_path": str(files["memory"]),
        },
    )


def build_synthesis_cortex(
    topic: str,
    plan_output: str,
    research_outputs: list[str],
    llm,
    model: str,
    files: dict[str, Any],
) -> Cortex:
    """Build the final synthesis stage Cortex."""
    analyst = create_analyst_agent(llm=llm, model=model)
    writer = create_writer_agent(llm=llm, model=model)

    research_briefs = []
    for index, output in enumerate(research_outputs, start=1):
        research_briefs.append(f"Track {index} Brief:\n{truncate_for_prompt(output, 3500)}")

    joined_briefs = "\n\n".join(research_briefs)
    synthesis_prompt_context = (
        f"Search Plan:\n{truncate_for_prompt(plan_output, 4000)}\n\nParallel Research Briefs:\n{joined_briefs}"
    )

    analysis_task = CortexTask(
        description=(
            f"Synthesize the collected evidence for '{topic}'.\n"
            "Return:\n"
            "1. strongest findings\n"
            "2. important disagreements or ambiguity across sources\n"
            "3. notable risks or implications\n"
            "4. what is still unverified\n"
            "5. the next best follow-up searches"
        ),
        expected_output="A markdown analytical brief separating strong evidence from uncertainty.",
        agent=analyst,
        prompt_context=synthesis_prompt_context,
        output_file=str(files["analysis"]),
        max_retries=2,
    )

    report_task = CortexTask(
        description=(
            f"Write the final deep-search report for '{topic}'.\n"
            "Use this structure:\n"
            "# Executive Summary\n"
            "# What We Know\n"
            "# Key Evidence\n"
            "# Contradictions And Uncertainty\n"
            "# Recommended Next Searches\n"
            "# Sources\n"
            "Use clear markdown and preserve source URLs as markdown links."
        ),
        expected_output="A production-ready markdown report with inline source links.",
        agent=writer,
        context=[analysis_task],
        prompt_context=f"Search Plan:\n{truncate_for_prompt(plan_output, 2000)}",
        output_file=str(files["report"]),
        max_retries=2,
    )

    return Cortex(
        agents=[analyst, writer],
        tasks=[analysis_task, report_task],
        llm=llm,
        process=ProcessType.SEQUENTIAL,
        verbose=True,
        model=model,
        cortex_name="DeepSearch Synthesis",
        memory_config={
            "enable_short_term": True,
            "enable_long_term": True,
            "enable_entity": True,
            "short_term_capacity": 100,
            "long_term_capacity": 1500,
            "persistence_path": str(files["memory"]),
        },
    )


def run_cortex(cortex: Cortex, stream: bool) -> CortexOutput:
    """Execute one Cortex stage, optionally with streaming."""
    if stream:
        buffer, thread = cortex.kickoff(use_streaming=True)
        for item in buffer.stream():
            if item is not None:
                stream_callback(item)
        thread.join()
        result = getattr(buffer, "cortex_output", None)
        if result is None:
            raise RuntimeError("Cortex finished without producing a final output.")
        return result
    return cortex.kickoff()


def write_metadata(
    topic: str,
    model: str,
    files: dict[str, Any],
    parallel_researchers: int,
    stage_results: dict[str, CortexOutput],
) -> None:
    """Write a small metadata manifest for the run."""
    payload = {
        "topic": topic,
        "model": model,
        "parallel_researchers": parallel_researchers,
        "generated_at": datetime.now().isoformat(),
        "llm_config": ENV_PY_LLM_CONFIG,
        "artifacts": serialize_paths(files),
        "stages": {
            stage_name: {
                "execution_time_seconds": result.execution_time,
                "task_count": len(result.task_outputs),
            }
            for stage_name, result in stage_results.items()
        },
        "total_execution_time_seconds": sum(result.execution_time for result in stage_results.values()),
    }
    files["metadata"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_deepsearch(
    topic: str,
    model: str = DEFAULT_MODEL,
    parallel_researchers: int = DEFAULT_PARALLEL_RESEARCHERS,
    stream: bool = True,
) -> tuple[CortexOutput, dict[str, Any], dict[str, CortexOutput]]:
    """Run the multi-stage deep-search pipeline."""
    if parallel_researchers < 1:
        raise ValueError("parallel_researchers must be at least 1")

    llm = build_llm_from_env_config()
    files = build_output_files(topic=topic, parallel_researchers=parallel_researchers)

    planning_cortex = build_planning_cortex(topic=topic, llm=llm, model=model, files=files)
    planning_result = run_cortex(planning_cortex, stream=stream)

    research_cortex = build_parallel_research_cortex(
        topic=topic,
        plan_output=planning_result.raw_output,
        llm=llm,
        model=model,
        files=files,
        parallel_researchers=parallel_researchers,
    )
    research_result = run_cortex(research_cortex, stream=stream)
    research_outputs = [task_output.output for task_output in research_result.task_outputs]

    synthesis_cortex = build_synthesis_cortex(
        topic=topic,
        plan_output=planning_result.raw_output,
        research_outputs=research_outputs,
        llm=llm,
        model=model,
        files=files,
    )
    synthesis_result = run_cortex(synthesis_cortex, stream=stream)

    stage_results = {
        "planning": planning_result,
        "research": research_result,
        "synthesis": synthesis_result,
    }
    write_metadata(
        topic=topic,
        model=model,
        files=files,
        parallel_researchers=parallel_researchers,
        stage_results=stage_results,
    )
    return synthesis_result, files, stage_results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a Cortex deep-search workflow.")
    parser.add_argument(
        "topic",
        nargs="?",
        default="latest AI coding agent developments",
        help="Topic to investigate.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to use. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "-n",
        "--parallel-researchers",
        type=int,
        default=DEFAULT_PARALLEL_RESEARCHERS,
        help=f"Number of parallel research agents. Default: {DEFAULT_PARALLEL_RESEARCHERS}",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output and only print the final summary.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    result, files, stage_results = run_deepsearch(
        topic=args.topic,
        model=args.model,
        parallel_researchers=args.parallel_researchers,
        stream=not args.no_stream,
    )

    total_time = sum(stage.execution_time for stage in stage_results.values())
    print("\nDeepSearch Cortex completed.")
    print(f"Parallel researchers: {args.parallel_researchers}")
    print(f"Execution time: {total_time:.2f}s")
    print(f"Final report: {files['report']}")
    print(f"Session dir: {files['session']}")
    print(f"Final stage time: {result.execution_time:.2f}s")


if __name__ == "__main__":
    main()
