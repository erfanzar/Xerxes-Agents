#!/usr/bin/env python3
"""
Scenario 3: Multi-Agent Collaboration System
Multiple specialized agents working together on complex tasks.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import openai

from calute import Agent, Calute
from calute.executors import EnhancedAgentOrchestrator, EnhancedFunctionExecutor
from calute.memory import MemoryStore, MemoryType
from calute.types import AgentSwitchTrigger

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR-KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", None),
)

# Shared memory for all agents
shared_memory = MemoryStore(
    max_short_term=200,
    max_working=50,
    enable_persistence=True,
    persistence_path=Path.home() / ".calute" / "multi_agent_memory",
)

# Global task queue for coordination
task_queue = []
completed_tasks = []


# ============ Research Agent Functions ============


def search_information(query: str) -> str:
    """Search for information on a topic."""
    # Simulate research
    research_data = {
        "machine learning": "ML is a subset of AI that enables systems to learn from data.",
        "python": "Python is a versatile programming language popular in data science.",
        "web development": "Web development involves creating websites and web applications.",
        "databases": "Databases store and organize data for efficient retrieval.",
    }

    query_lower = query.lower()
    results = []
    for topic, info in research_data.items():
        if any(word in query_lower for word in topic.split()):
            results.append(f"{topic}: {info}")

    # Store research in memory
    shared_memory.add_memory(
        content=f"Research on '{query}': {len(results)} results found",
        memory_type=MemoryType.SEMANTIC,
        agent_id="research_agent",
        tags=["research", "information", query.lower()],
        importance_score=0.7,
    )

    if results:
        return "Research findings:\n" + "\n".join(f"• {r}" for r in results)
    return f"No specific information found on '{query}'"


def compile_report(topic: str) -> str:
    """Compile a report from gathered information."""
    # Retrieve relevant memories
    memories = shared_memory.retrieve_memories(tags=["research", topic.lower()], limit=10)

    report = f"📄 Report on: {topic}\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    if memories:
        report += "Key Findings:\n"
        for mem in memories:
            report += f"• {mem.content}\n"
    else:
        report += "No prior research found on this topic.\n"

    # Store report
    shared_memory.add_memory(
        content=f"Report compiled on '{topic}'",
        memory_type=MemoryType.LONG_TERM,
        agent_id="research_agent",
        tags=["report", topic.lower()],
        importance_score=0.8,
    )

    return report


# ============ Planning Agent Functions ============


def create_project_plan(project_name: str, requirements: str) -> str:
    """Create a project plan with tasks and milestones."""
    # Parse requirements
    req_list = requirements.split(",") if "," in requirements else [requirements]

    plan = {
        "project": project_name,
        "created": datetime.now().isoformat(),
        "phases": [
            {
                "name": "Research & Analysis",
                "tasks": [f"Research {req.strip()}" for req in req_list[:2]],
                "duration": "2 days",
            },
            {
                "name": "Design & Architecture",
                "tasks": ["Create system design", "Define data models", "Plan API structure"],
                "duration": "3 days",
            },
            {"name": "Implementation", "tasks": [f"Implement {req.strip()}" for req in req_list], "duration": "5 days"},
            {
                "name": "Testing & Deployment",
                "tasks": ["Write tests", "Perform QA", "Deploy to production"],
                "duration": "2 days",
            },
        ],
    }

    # Add tasks to queue
    for phase in plan["phases"]:
        for task in phase["tasks"]:
            task_queue.append(
                {
                    "id": f"{project_name}_{len(task_queue)}",
                    "task": task,
                    "phase": phase["name"],
                    "status": "pending",
                    "assigned_to": None,
                }
            )

    # Store plan
    shared_memory.add_memory(
        content=f"Project plan created: {project_name} with {len(task_queue)} tasks",
        memory_type=MemoryType.PROCEDURAL,
        agent_id="planning_agent",
        tags=["plan", "project", project_name.lower()],
        importance_score=0.9,
    )

    return f"📋 Project Plan:\n{json.dumps(plan, indent=2)}\n\nTotal tasks created: {len(task_queue)}"


def assign_tasks() -> str:
    """Assign pending tasks to appropriate agents."""
    assignments = []

    for task in task_queue:
        if task["status"] == "pending":
            # Determine best agent for task
            if "research" in task["task"].lower() or "analyze" in task["task"].lower():
                task["assigned_to"] = "research_agent"
            elif "implement" in task["task"].lower() or "code" in task["task"].lower():
                task["assigned_to"] = "implementation_agent"
            elif "test" in task["task"].lower() or "qa" in task["task"].lower():
                task["assigned_to"] = "qa_agent"
            else:
                task["assigned_to"] = "research_agent"  # Default

            task["status"] = "assigned"
            assignments.append(f"{task['task']} → {task['assigned_to']}")

    if assignments:
        result = "📌 Task Assignments:\n"
        for assignment in assignments:
            result += f"• {assignment}\n"
    else:
        result = "No pending tasks to assign."

    return result


def track_progress() -> str:
    """Track project progress and task completion."""
    total_tasks = len(task_queue) + len(completed_tasks)
    if total_tasks == 0:
        return "No tasks to track."

    pending = sum(1 for t in task_queue if t["status"] == "pending")
    assigned = sum(1 for t in task_queue if t["status"] == "assigned")
    in_progress = sum(1 for t in task_queue if t["status"] == "in_progress")
    completed = len(completed_tasks)

    progress = (completed / total_tasks) * 100 if total_tasks > 0 else 0

    result = f"""📊 Project Progress Report:

Total Tasks: {total_tasks}
• Pending: {pending}
• Assigned: {assigned}
• In Progress: {in_progress}
• Completed: {completed}

Overall Progress: {progress:.1f}%
{"█" * int(progress / 5)}{"░" * (20 - int(progress / 5))} {progress:.0f}%
"""

    # Recent completions
    if completed_tasks:
        result += "\n🎯 Recently Completed:\n"
        for task in completed_tasks[-3:]:
            result += f"• {task['task']} ✓\n"

    return result


# ============ Implementation Agent Functions ============


def implement_feature(feature_name: str, specifications: str = "") -> str:
    """Implement a software feature."""
    # Simulate implementation
    code_template = f"""
# Implementation of {feature_name}
# Specifications: {specifications}

class {feature_name.replace(" ", "")}:
    def __init__(self):
        self.name = "{feature_name}"
        self.created_at = "{datetime.now().isoformat()}"

    def execute(self):
        # Implementation logic here
        return f"Executing {feature_name}"

    def validate(self):
        # Validation logic
        return True
"""

    # Mark task as complete
    for task in task_queue:
        if feature_name.lower() in task["task"].lower() and task["status"] == "assigned":
            task["status"] = "completed"
            completed_tasks.append(task)
            task_queue.remove(task)
            break

    # Store implementation
    shared_memory.add_memory(
        content=f"Implemented feature: {feature_name}",
        memory_type=MemoryType.PROCEDURAL,
        agent_id="implementation_agent",
        tags=["implementation", "code", feature_name.lower()],
        importance_score=0.8,
    )

    return f"✅ Feature Implemented:\n```python{code_template}```"


def review_code(code_or_feature: str) -> str:
    """Review code or implementation for quality."""
    reviews = []

    # Simulate code review
    if "class" in code_or_feature.lower():
        reviews.append("✓ Class structure looks good")
    if "def " in code_or_feature.lower():
        reviews.append("✓ Methods are well-defined")
    if "validate" in code_or_feature.lower():
        reviews.append("✓ Validation logic present")

    issues = []
    if "TODO" in code_or_feature or "FIXME" in code_or_feature:
        issues.append("⚠️ Contains TODO/FIXME comments")
    if "test" not in code_or_feature.lower():
        issues.append("⚠️ Consider adding unit tests")

    result = "🔍 Code Review Results:\n"
    if reviews:
        result += "\nPositive:\n" + "\n".join(reviews)
    if issues:
        result += "\n\nNeeds Attention:\n" + "\n".join(issues)

    result += f"\n\nOverall Score: {len(reviews)}/{len(reviews) + len(issues)} ⭐"

    return result


# ============ Coordinator Function ============


def coordinate_agents(task_description: str) -> str:
    """Coordinate multiple agents to complete a complex task."""
    # Parse task
    subtasks = []

    if "and" in task_description:
        parts = task_description.split("and")
        for part in parts:
            subtasks.append(part.strip())
    else:
        subtasks = [task_description]

    coordination_plan = {
        "main_task": task_description,
        "subtasks": subtasks,
        "agent_assignments": [],
        "estimated_time": f"{len(subtasks) * 2} hours",
    }

    # Assign subtasks
    for subtask in subtasks:
        if "research" in subtask.lower():
            coordination_plan["agent_assignments"].append({"task": subtask, "agent": "research_agent"})
        elif "plan" in subtask.lower():
            coordination_plan["agent_assignments"].append({"task": subtask, "agent": "planning_agent"})
        elif "implement" in subtask.lower() or "build" in subtask.lower():
            coordination_plan["agent_assignments"].append({"task": subtask, "agent": "implementation_agent"})
        else:
            coordination_plan["agent_assignments"].append({"task": subtask, "agent": "research_agent"})

    result = f"🎯 Coordination Plan:\n{json.dumps(coordination_plan, indent=2)}"

    # Store coordination
    shared_memory.add_memory(
        content=f"Coordinated task: {task_description}",
        memory_type=MemoryType.EPISODIC,
        agent_id="coordinator",
        tags=["coordination", "multi-agent"],
        importance_score=0.9,
    )

    return result


async def main():
    """Run the multi-agent collaboration scenario."""
    print("=" * 60)
    print("🤝 MULTI-AGENT COLLABORATION SYSTEM")
    print("=" * 60)
    print()

    # Create specialized agents
    research_agent = Agent(
        id="research_agent",
        name="Research Specialist",
        model="gpt-4o",
        instructions="You are a research specialist. Find and compile information.",
        functions=[search_information, compile_report],
        max_tokens=500,
        temperature=0.5,
    )

    planning_agent = Agent(
        id="planning_agent",
        name="Project Planner",
        model="gpt-4o",
        instructions="You are a project planner. Create plans, assign tasks, and track progress.",
        functions=[create_project_plan, assign_tasks, track_progress],
        max_tokens=500,
        temperature=0.4,
    )

    implementation_agent = Agent(
        id="implementation_agent",
        name="Implementation Expert",
        model="gpt-4o",
        instructions="You implement features and review code quality.",
        functions=[implement_feature, review_code],
        max_tokens=500,
        temperature=0.3,
    )

    coordinator_agent = Agent(
        id="coordinator",
        name="Team Coordinator",
        model="gpt-4o",
        instructions="""You coordinate between different agents to accomplish complex tasks.
        Break down tasks and assign them to the right specialists.""",
        functions=[coordinate_agents],
        max_tokens=500,
        temperature=0.5,
    )

    # Set up orchestrator
    orchestrator = EnhancedAgentOrchestrator(max_agents=10, enable_metrics=True)

    # Register all agents
    await orchestrator.register_agent(research_agent)
    await orchestrator.register_agent(planning_agent)
    await orchestrator.register_agent(implementation_agent)
    await orchestrator.register_agent(coordinator_agent)

    print(f"✅ Registered {len(orchestrator.agents)} specialized agents\n")

    # Create executor
    executor = EnhancedFunctionExecutor(orchestrator=orchestrator, default_timeout=30.0, max_concurrent_executions=4)

    # Initialize Calute
    calute = Calute(client, enable_memory=True)
    calute.memory = shared_memory

    # Register agents with Calute
    for agent in [research_agent, planning_agent, implementation_agent, coordinator_agent]:
        calute.register_agent(agent)

    # Complex multi-agent task
    print("🎯 COMPLEX TASK: Build a web application for task management\n")

    # Step 1: Coordinator breaks down the task using enhanced executor
    print("Step 1: Coordinator analyzes the task...")

    from calute.types import FunctionCallStrategy, RequestFunctionCall

    # Create function call for coordination
    coordination_call = RequestFunctionCall(
        name="coordinate_agents",
        arguments={
            "task_description": "Research task management systems and plan the architecture and implement core features"
        },
        id="coord_1",
    )

    try:
        results = await executor.execute_function_calls(
            calls=[coordination_call], strategy=FunctionCallStrategy.SEQUENTIAL, agent=coordinator_agent
        )
        coordination_result = results[0].result if hasattr(results[0], "result") else str(results[0])
        print(coordination_result)
    except Exception as e:
        print(f"Error during coordination: {e}")
        # Fallback to direct call
        coordination_result = coordinate_agents(
            "Research task management systems and plan the architecture and implement core features"
        )
        print(coordination_result)

    print()

    # Step 2: Research phase using enhanced executor
    print("Step 2: Research Agent gathering information...")

    # Create research function calls
    research_calls = [
        RequestFunctionCall(name="search_information", arguments={"query": topic}, id=f"research_{i}")
        for i, topic in enumerate(["task management", "web development", "databases"])
    ]

    # Add report compilation
    research_calls.append(
        RequestFunctionCall(name="compile_report", arguments={"topic": "Task Management System"}, id="report_1")
    )

    try:
        results = await executor.execute_function_calls(
            calls=research_calls, strategy=FunctionCallStrategy.PARALLEL, agent=research_agent
        )

        # Display research results
        for _, topic in enumerate(["task management", "web development", "databases"]):
            print(f"  • Researched: {topic}")

        # Display report (last result)
        report = results[-1].result if hasattr(results[-1], "result") else str(results[-1])
        print(report)
    except Exception as e:
        print(f"Error during research: {e}")
        # Fallback to direct calls
        for topic in ["task management", "web development", "databases"]:
            result = search_information(topic)
            print(f"  • Researched: {topic}")
            print(f"    Result: {result[:100]}..." if len(result) > 100 else f"    Result: {result}")
        report = compile_report("Task Management System")
        print(report)

    print()

    # Step 3: Planning phase using enhanced executor
    print("Step 3: Planning Agent creates project plan...")

    # Create planning function calls
    planning_calls = [
        RequestFunctionCall(
            name="create_project_plan",
            arguments={
                "project_name": "TaskManager",
                "requirements": "user authentication, task creation, task assignment, progress tracking, notifications",
            },
            id="plan_1",
        ),
        RequestFunctionCall(name="assign_tasks", arguments={}, id="assign_1"),
    ]

    try:
        results = await executor.execute_function_calls(
            calls=planning_calls, strategy=FunctionCallStrategy.SEQUENTIAL, agent=planning_agent
        )

        # Display plan
        plan = results[0].result if hasattr(results[0], "result") else str(results[0])
        print(plan)
        print()

        # Display assignments
        print("Step 4: Assigning tasks to agents...")
        assignments = results[1].result if hasattr(results[1], "result") else str(results[1])
        print(assignments)
    except Exception as e:
        print(f"Error during planning: {e}")
        # Fallback to direct calls
        plan = create_project_plan(
            "TaskManager", "user authentication, task creation, task assignment, progress tracking, notifications"
        )
        print(plan)
        print()
        print("Step 4: Assigning tasks to agents...")
        assignments = assign_tasks()
        print(assignments)

    print()

    # Step 5: Implementation phase using enhanced executor
    print("Step 5: Implementation Agent working on features...")
    features = ["User Authentication", "Task Creation", "Progress Tracking"]

    # Create implementation and review calls for each feature
    implementation_calls = []
    for feature in features:
        implementation_calls.append(
            RequestFunctionCall(
                name="implement_feature",
                arguments={"feature_name": feature, "specifications": "Core functionality required"},
                id=f"impl_{feature.replace(' ', '_').lower()}",
            )
        )

    try:
        # Execute implementations in parallel
        impl_results = await executor.execute_function_calls(
            calls=implementation_calls, strategy=FunctionCallStrategy.PARALLEL, agent=implementation_agent
        )

        # Review each implementation
        for i, (feature, impl_result) in enumerate(zip(features, impl_results, strict=False)):
            print(f"\n  Working on: {feature}")
            print(f"  ✓ Implemented {feature}")

            # Create review call
            implementation = impl_result.result if hasattr(impl_result, "result") else str(impl_result)
            review_call = RequestFunctionCall(
                name="review_code", arguments={"code_or_feature": implementation}, id=f"review_{i}"
            )

            # Execute review
            review_results = await executor.execute_function_calls(
                calls=[review_call], strategy=FunctionCallStrategy.SEQUENTIAL, agent=implementation_agent
            )

            review = review_results[0].result if hasattr(review_results[0], "result") else str(review_results[0])
            print(f"  Review: {review[:100]}...")

            await asyncio.sleep(0.5)  # Simulate work time
    except Exception as e:
        print(f"Error during implementation: {e}")
        # Fallback to direct calls
        for feature in features:
            print(f"\n  Working on: {feature}")
            implementation = implement_feature(feature, "Core functionality required")
            print(f"  ✓ Implemented {feature}")
            review = review_code(implementation)
            print(f"  Review: {review[:100]}...")
            await asyncio.sleep(0.5)

    print()

    # Step 6: Progress tracking using enhanced executor
    print("Step 6: Tracking overall progress...")

    progress_call = RequestFunctionCall(name="track_progress", arguments={}, id="progress_1")

    try:
        results = await executor.execute_function_calls(
            calls=[progress_call], strategy=FunctionCallStrategy.SEQUENTIAL, agent=planning_agent
        )
        progress = results[0].result if hasattr(results[0], "result") else str(results[0])
        print(progress)
    except Exception as e:
        print(f"Error tracking progress: {e}")
        # Fallback to direct call
        progress = track_progress()
        print(progress)

    print()

    # Step 7: Agent switching demonstration
    print("Step 7: Dynamic agent switching based on task...")

    # Define switch trigger
    def task_based_switch(context: dict, agents: dict, current_agent_id: str) -> str | None:
        """Switch agent based on task type."""
        task = context.get("current_task", "")

        if "research" in task.lower():
            return "research_agent"
        elif "plan" in task.lower():
            return "planning_agent"
        elif "implement" in task.lower():
            return "implementation_agent"

        return current_agent_id

    orchestrator.register_switch_trigger(AgentSwitchTrigger.CONTEXT_BASED, task_based_switch)

    # Demonstrate switching
    tasks = [
        {"current_task": "research best practices", "expected_agent": "research_agent"},
        {"current_task": "plan deployment strategy", "expected_agent": "planning_agent"},
        {"current_task": "implement caching layer", "expected_agent": "implementation_agent"},
    ]

    for task_context in tasks:
        target_agent = orchestrator.should_switch_agent(task_context)
        if target_agent:
            await orchestrator.switch_agent(target_agent, f"Task requires {target_agent}")
            print(f"  → Switched to {target_agent} for: {task_context['current_task']}")

    print()

    # Show collaboration metrics
    print("=" * 60)
    print("📊 COLLABORATION METRICS")
    print("=" * 60)

    # Memory statistics
    stats = shared_memory.get_statistics()
    print("\n📝 Shared Memory:")
    print(f"  • Total memories: {stats['total_memories']}")
    print(f"  • By type: {stats['by_type']}")

    # Agent activity
    print("\n👥 Agent Activity:")
    for agent_id in ["research_agent", "planning_agent", "implementation_agent", "coordinator"]:
        agent_memories = shared_memory.retrieve_memories(agent_id=agent_id, limit=100)
        print(f"  • {agent_id}: {len(agent_memories)} actions")

    # Execution history
    if orchestrator.execution_history:
        print(f"\n🔄 Agent Switches: {len(orchestrator.execution_history)}")
        for event in orchestrator.execution_history[-3:]:
            print(f"  • {event['from']} → {event['to']}: {event.get('reason', 'N/A')}")

    # Enhanced executor metrics
    if hasattr(executor, "function_registry"):
        print("\n⚡ Enhanced Executor Metrics:")
        for func_name in [
            "coordinate_agents",
            "search_information",
            "create_project_plan",
            "implement_feature",
            "track_progress",
        ]:
            metrics = executor.function_registry.get_metrics(func_name)
            if metrics and metrics.total_calls > 0:
                print(
                    f"  • {func_name}: {metrics.total_calls} calls, "
                    f"{metrics.successful_calls / metrics.total_calls:.0%} success rate"
                )

    # Task completion
    print("\n✅ Task Completion:")
    print(f"  • Total tasks: {len(task_queue) + len(completed_tasks)}")
    print(f"  • Completed: {len(completed_tasks)}")
    print(f"  • Success rate: {len(completed_tasks) / (len(task_queue) + len(completed_tasks)) * 100:.0f}%")

    # Save shared memory
    shared_memory.save()
    print("\n💾 Collaboration data saved!")

    print("\n✅ Multi-agent collaboration scenario completed!")


if __name__ == "__main__":
    asyncio.run(main())
