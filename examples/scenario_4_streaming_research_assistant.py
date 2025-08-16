#!/usr/bin/env python3
"""
Scenario 4: Real-time Streaming Research Assistant
An advanced research assistant that streams responses and builds knowledge dynamically.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import openai

from calute import Agent, Calute, FunctionExecutionComplete, MessagesHistory, StreamChunk, UserMessage
from calute.memory import EnhancedMemoryStore, MemoryType
from calute.tools import DuckDuckGoSearch, ReadFile, WriteFile

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR-KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", None),
)

# Research memory with vector search capability
research_memory = EnhancedMemoryStore(
    max_short_term=500,
    max_long_term=10000,
    enable_persistence=True,
    persistence_path=Path.home() / ".calute" / "research_assistant_memory",
    enable_vector_search=False,  # Set to True if you have embeddings
)

# Knowledge base
knowledge_base = {}


def extract_key_points(text: str, max_points: int = 5) -> str:
    """Extract key points from text."""
    # Simple extraction based on sentences
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return "No key points found."

    # Take first max_points sentences as key points
    key_points = sentences[:max_points]

    # Store in memory
    for point in key_points:
        research_memory.add_memory(
            content=point,
            memory_type=MemoryType.SEMANTIC,
            agent_id="research_assistant",
            tags=["key_point", "extraction"],
            importance_score=0.7,
        )

    result = "📌 Key Points:\n"
    for i, point in enumerate(key_points, 1):
        result += f"{i}. {point}\n"

    return result


def build_knowledge_graph(topic: str, facts: list[str]) -> str:
    """Build a knowledge graph from research facts."""
    global knowledge_base

    if topic not in knowledge_base:
        knowledge_base[topic] = {"facts": [], "connections": [], "sources": [], "timestamp": datetime.now().isoformat()}

    # Add facts
    for fact in facts:
        if fact not in knowledge_base[topic]["facts"]:
            knowledge_base[topic]["facts"].append(fact)

            # Store in long-term memory
            research_memory.add_memory(
                content=f"{topic}: {fact}",
                memory_type=MemoryType.LONG_TERM,
                agent_id="research_assistant",
                tags=["knowledge", topic.lower(), "fact"],
                importance_score=0.8,
            )

    # Find connections to other topics
    for other_topic in knowledge_base:
        if other_topic != topic:
            # Simple connection detection
            topic_words = set(topic.lower().split())
            other_words = set(other_topic.lower().split())
            if topic_words & other_words:
                connection = f"{topic} ↔ {other_topic}"
                if connection not in knowledge_base[topic]["connections"]:
                    knowledge_base[topic]["connections"].append(connection)

    result = f"📊 Knowledge Graph for '{topic}':\n"
    result += f"  • Facts: {len(knowledge_base[topic]['facts'])}\n"
    result += f"  • Connections: {len(knowledge_base[topic]['connections'])}\n"

    if knowledge_base[topic]["connections"]:
        result += "  • Related topics: " + ", ".join(c.split(" ↔ ")[1] for c in knowledge_base[topic]["connections"][:3])

    return result


def synthesize_research(topics: list[str]) -> str:
    """Synthesize research from multiple topics."""
    synthesis = {"topics": topics, "total_facts": 0, "common_themes": [], "insights": []}

    all_facts = []
    word_frequency = {}

    for topic in topics:
        # Retrieve memories about this topic
        memories = research_memory.retrieve_memories(
            tags=[topic.lower()], memory_types=[MemoryType.LONG_TERM, MemoryType.SEMANTIC], limit=10
        )

        for mem in memories:
            all_facts.append(mem.content)
            # Count word frequency
            words = mem.content.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    word_frequency[word] = word_frequency.get(word, 0) + 1

    synthesis["total_facts"] = len(all_facts)

    # Find common themes
    common_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
    synthesis["common_themes"] = [word for word, _ in common_words]

    # Generate insights
    if len(topics) > 1:
        synthesis["insights"].append(f"Found {len(all_facts)} facts across {len(topics)} topics")
        if common_words:
            synthesis["insights"].append(f"Common themes: {', '.join(synthesis['common_themes'])}")

    # Create synthesis report
    result = "🔬 Research Synthesis:\n"
    result += f"Topics analyzed: {', '.join(topics)}\n"
    result += f"Total facts gathered: {synthesis['total_facts']}\n"

    if synthesis["common_themes"]:
        result += f"Common themes: {', '.join(synthesis['common_themes'])}\n"

    if synthesis["insights"]:
        result += "\n💡 Insights:\n"
        for insight in synthesis["insights"]:
            result += f"  • {insight}\n"

    # Store synthesis
    research_memory.add_memory(
        content=f"Synthesis of {len(topics)} topics: {synthesis['total_facts']} facts",
        memory_type=MemoryType.SEMANTIC,
        agent_id="research_assistant",
        tags=["synthesis"] + [t.lower() for t in topics],
        importance_score=0.9,
    )

    return result


def create_research_outline(topic: str, depth: str = "moderate") -> str:
    """Create a research outline for a topic."""
    depth_levels = {"basic": 3, "moderate": 5, "comprehensive": 8}

    num_sections = depth_levels.get(depth, 5)

    outline = {
        "title": f"Research Outline: {topic}",
        "created": datetime.now().isoformat(),
        "depth": depth,
        "sections": [],
    }

    # Generate sections based on topic
    base_sections = [
        {"title": "Introduction", "subsections": ["Definition", "Historical Context", "Importance"]},
        {"title": "Core Concepts", "subsections": ["Fundamental Principles", "Key Terminology"]},
        {"title": "Current State", "subsections": ["Recent Developments", "Leading Research"]},
        {"title": "Applications", "subsections": ["Practical Uses", "Case Studies"]},
        {"title": "Challenges", "subsections": ["Current Limitations", "Open Problems"]},
        {"title": "Future Directions", "subsections": ["Emerging Trends", "Predictions"]},
        {"title": "Related Topics", "subsections": ["Connected Fields", "Interdisciplinary Aspects"]},
        {"title": "Conclusion", "subsections": ["Summary", "Key Takeaways"]},
    ]

    outline["sections"] = base_sections[:num_sections]

    # Format outline
    result = f"📋 {outline['title']}\n"
    result += f"Depth: {depth}\n\n"

    for i, section in enumerate(outline["sections"], 1):
        result += f"{i}. {section['title']}\n"
        for subsection in section["subsections"]:
            result += f"   • {subsection}\n"

    # Store outline
    research_memory.add_memory(
        content=f"Research outline created for '{topic}' with {len(outline['sections'])} sections",
        memory_type=MemoryType.PROCEDURAL,
        agent_id="research_assistant",
        tags=["outline", "research_plan", topic.lower()],
        importance_score=0.7,
    )

    return result


def track_research_progress(topic: str) -> str:
    """Track progress on research topics."""
    # Retrieve all memories related to the topic
    memories = research_memory.retrieve_memories(tags=[topic.lower()], limit=100)

    progress = {"topic": topic, "total_items": len(memories), "by_type": {}, "timeline": [], "completeness": 0}

    # Analyze memories
    for mem in memories:
        mem_type = mem.memory_type.value
        progress["by_type"][mem_type] = progress["by_type"].get(mem_type, 0) + 1

    # Calculate completeness (simple heuristic)
    expected_items = 20  # Expected number of research items
    progress["completeness"] = min(100, (len(memories) / expected_items) * 100)

    # Format progress report
    result = f"📈 Research Progress: {topic}\n"
    result += f"{'═' * 40}\n"
    result += f"Completeness: {progress['completeness']:.0f}%\n"
    result += f"{'█' * int(progress['completeness'] / 5)}{'░' * (20 - int(progress['completeness'] / 5))}\n\n"

    result += f"Total research items: {progress['total_items']}\n"

    if progress["by_type"]:
        result += "By category:\n"
        for mem_type, count in progress["by_type"].items():
            result += f"  • {mem_type}: {count}\n"

    # Recommendations
    result += "\n📝 Recommendations:\n"
    if progress["completeness"] < 30:
        result += "  • Need more initial research\n"
    elif progress["completeness"] < 60:
        result += "  • Good start, continue gathering information\n"
    elif progress["completeness"] < 90:
        result += "  • Nearly complete, focus on synthesis\n"
    else:
        result += "  • Research complete, ready for final report\n"

    return result


async def stream_research_response(calute, agent, prompt, messages):
    """Stream a research response with real-time updates."""
    print("\n🔄 Streaming response...")
    print("-" * 40)

    full_response = ""
    function_calls = []

    try:
        stream = await calute.create_response(
            prompt=prompt,
            messages=messages,
            agent_id=agent.id,
            stream=True,
            apply_functions=True,
            use_instructed_prompt=True,
        )

        async for chunk in stream:
            if isinstance(chunk, StreamChunk) and chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content

            elif isinstance(chunk, FunctionExecutionComplete):
                function_calls.append(
                    {"function": chunk.function_name, "status": chunk.status, "timestamp": datetime.now().isoformat()}
                )
                print(f"\n[Function: {chunk.function_name} ✓]", end="", flush=True)

        print("\n" + "-" * 40)

    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
        full_response = "Error during streaming"

    # Store streamed response
    if full_response:
        research_memory.add_memory(
            content=f"Research response: {full_response[:200]}...",
            memory_type=MemoryType.EPISODIC,
            agent_id="research_assistant",
            tags=["response", "stream"],
            importance_score=0.6,
        )

    return full_response, function_calls


async def main():
    """Run the streaming research assistant scenario."""
    print("=" * 60)
    print("🔬 REAL-TIME STREAMING RESEARCH ASSISTANT")
    print("=" * 60)
    print()

    # Create research assistant agent
    agent = Agent(
        id="research_assistant",
        name="Advanced Research Assistant",
        model="gpt-4o",
        instructions="""You are an advanced research assistant with real-time streaming capabilities.
        Your role is to:
        1. Conduct thorough research on any topic
        2. Extract and synthesize key information
        3. Build knowledge graphs and connections
        4. Track research progress
        5. Create comprehensive outlines
        6. Stream responses for real-time interaction

        Use your tools to gather, analyze, and present information effectively.
        Be thorough, accurate, and organize information clearly.""",
        functions=[
            extract_key_points,
            build_knowledge_graph,
            synthesize_research,
            create_research_outline,
            track_research_progress,
            DuckDuckGoSearch,  # Real web search if available
            WriteFile,  # Save research results
            ReadFile,  # Read saved research
        ],
        max_tokens=1000,
        temperature=0.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    # Initialize Calute
    calute = Calute(client, enable_memory=True)
    calute.memory = research_memory
    calute.register_agent(agent)

    # Research topics
    research_topics = ["artificial intelligence", "machine learning applications", "neural networks"]

    messages = MessagesHistory(messages=[])

    print("📚 Starting Research Session\n")

    # Phase 1: Initial Research
    print("=" * 40)
    print("PHASE 1: INITIAL RESEARCH")
    print("=" * 40)

    for topic in research_topics:
        print(f"\n🔍 Researching: {topic}")

        # Create outline
        outline = create_research_outline(topic, "moderate")
        print(outline[:200] + "...\n")

        # Build knowledge
        sample_facts = [
            f"{topic} is a key technology",
            f"{topic} has multiple applications",
            f"{topic} is evolving rapidly",
        ]
        knowledge_result = build_knowledge_graph(topic, sample_facts)
        print(knowledge_result)

        # Track progress
        progress = track_research_progress(topic)
        print(progress)

        await asyncio.sleep(0.5)

    # Phase 2: Streaming Research
    print("\n" + "=" * 40)
    print("PHASE 2: STREAMING RESEARCH")
    print("=" * 40)

    research_queries = [
        "What are the latest developments in artificial intelligence?",
        "How do neural networks learn from data?",
        "What are the practical applications of machine learning?",
    ]

    for query in research_queries:
        print(f"\n❓ Query: {query}")

        messages.messages.append(UserMessage(content=query))

        # Stream response
        response, functions_used = await stream_research_response(calute, agent, query, messages)

        if functions_used:
            print(f"\n📊 Functions used: {len(functions_used)}")
            for func in functions_used:
                print(f"  • {func['function']}")

        # Extract key points from response
        if len(response) > 100:
            key_points = extract_key_points(response, 3)
            print(f"\n{key_points}")

        await asyncio.sleep(1)

    # Phase 3: Synthesis
    print("\n" + "=" * 40)
    print("PHASE 3: RESEARCH SYNTHESIS")
    print("=" * 40)

    synthesis = synthesize_research(research_topics)
    print(synthesis)

    # Phase 4: Knowledge Base Summary
    print("\n" + "=" * 40)
    print("PHASE 4: KNOWLEDGE BASE")
    print("=" * 40)

    print("\n📚 Built Knowledge Base:")
    for topic, data in knowledge_base.items():
        print(f"\n{topic}:")
        print(f"  • Facts: {len(data['facts'])}")
        print(f"  • Connections: {len(data['connections'])}")
        if data["facts"]:
            print(f"  • Sample: {data['facts'][0][:50]}...")

    # Show memory statistics
    print("\n" + "=" * 40)
    print("📊 RESEARCH STATISTICS")
    print("=" * 40)

    stats = research_memory.get_statistics()
    print("\nMemory Statistics:")
    print(f"  • Total memories: {stats['total_memories']}")
    print(f"  • Memory distribution: {stats['by_type']}")
    print(f"  • Cache performance: {stats['cache_hit_rate']:.1%} hit rate")

    # Analyze research patterns

    if isinstance(research_memory, EnhancedMemoryStore):
        # Get most important memories
        important_memories = research_memory.retrieve_memories(min_importance=0.8, limit=5)

        if important_memories:
            print("\n🌟 Most Important Findings:")
            for mem in important_memories:
                print(f"  • {mem.content[:80]}...")

    # Save research results
    research_output = {
        "session_date": datetime.now().isoformat(),
        "topics_researched": research_topics,
        "total_queries": len(research_queries),
        "knowledge_base_size": sum(len(data["facts"]) for data in knowledge_base.values()),
        "memory_stats": stats,
    }

    output_path = Path.home() / ".calute" / "research_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(research_output, f, indent=2, default=str)

    print(f"\n💾 Research saved to {output_path}")

    # Save memory
    research_memory.save()
    print("💾 Research memory persisted")

    print("\n✅ Streaming research assistant scenario completed!")


if __name__ == "__main__":
    asyncio.run(main())
