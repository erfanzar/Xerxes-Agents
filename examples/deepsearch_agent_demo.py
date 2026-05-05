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
"""
DeepSearch Agent Demo - Advanced Information Discovery and Analysis

This demo showcases a powerful DeepSearch agent that combines multiple search strategies,
intelligent analysis, and comprehensive reporting capabilities. The agent can:

1. Perform multi-layered web searches with different strategies
2. Analyze and synthesize information from multiple sources
3. Extract entities, patterns, and insights
4. Create comprehensive research reports
5. Build knowledge graphs and connections
6. Track search progress and optimize strategies

Features demonstrated:
- Multi-source search (web, academic, news, social)
- Content analysis and entity extraction
- Intelligent query reformulation
- Real-time progress tracking
- Knowledge synthesis and visualization
- Automated report generation
"""

import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openai
from xerxes import Agent, MessagesHistory, UserMessage, Xerxes
from xerxes.memory import MemoryStore, MemoryType
from xerxes.tools import (
    DataConverter,
    EntityExtractor,
    GoogleSearch,
    JSONProcessor,
    ReadFile,
    StatisticalAnalyzer,
    TextClassifier,
    TextProcessor,
    TextSimilarity,
    TextSummarizer,
    URLAnalyzer,
    WebScraper,
    WriteFile,
)

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "YOUR-KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://137.184.183.6:11558/v1"),
)

# Enhanced memory for search intelligence
search_memory = MemoryStore(
    max_short_term=1000,
    max_long_term=20000,
    enable_persistence=True,
    persistence_path=Path.home() / ".xerxes" / "deepsearch_memory",
    enable_vector_search=False,  # Set to True if you have embeddings
)

# Global search state
search_session = {
    "queries": [],
    "results": [],
    "insights": {},
    "knowledge_graph": {},
    "search_strategies": [],
    "performance_metrics": {},
    "start_time": None,
}


def initialize_search_session(topic: str) -> str:
    """Initialize a new search session for a topic."""
    global search_session

    search_session = {
        "topic": topic,
        "queries": [],
        "results": [],
        "insights": {},
        "knowledge_graph": {},
        "search_strategies": ["broad", "specific", "academic", "news", "trends"],
        "performance_metrics": {"queries_executed": 0, "results_found": 0, "processing_time": 0},
        "start_time": datetime.now(),
        "session_id": f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    # Store session initialization
    search_memory.add_memory(
        content=f"DeepSearch session initialized for topic: {topic}",
        memory_type=MemoryType.EPISODIC,
        agent_id="deepsearch_agent",
        tags=["session", "initialization", topic.lower()],
        importance_score=0.8,
    )

    return (
        f"🔍 DeepSearch initialized for: '{topic}'\n"
        f"📊 Session ID: {search_session['session_id']}\n"
        f"⏰ Started: {search_session['start_time'].strftime('%Y-%m-%d %H:%M:%S')}"
    )


def generate_search_queries(base_topic: str, strategy: str = "comprehensive", max_queries: int = 8) -> str:
    """Generate intelligent search queries using different strategies."""

    strategies = {
        "broad": [
            f"{base_topic} overview",
            f"what is {base_topic}",
            f"{base_topic} introduction guide",
            f"{base_topic} basics fundamentals",
        ],
        "specific": [
            f"{base_topic} technical details",
            f"{base_topic} advanced concepts",
            f"{base_topic} implementation methods",
            f"{base_topic} best practices",
        ],
        "academic": [
            f"{base_topic} research papers",
            f"{base_topic} academic studies",
            f"{base_topic} scientific literature",
            f"{base_topic} peer reviewed",
        ],
        "news": [
            f"{base_topic} latest news",
            f"{base_topic} recent developments",
            f"{base_topic} current trends 2024",
            f"{base_topic} breaking news",
        ],
        "trends": [
            f"{base_topic} future outlook",
            f"{base_topic} predictions",
            f"{base_topic} emerging trends",
            f"{base_topic} market analysis",
        ],
        "problems": [
            f"{base_topic} challenges problems",
            f"{base_topic} limitations issues",
            f"{base_topic} solutions fixes",
            f"{base_topic} troubleshooting",
        ],
        "applications": [
            f"{base_topic} use cases applications",
            f"{base_topic} real world examples",
            f"{base_topic} practical applications",
            f"{base_topic} case studies",
        ],
    }

    if strategy == "comprehensive":
        # Mix queries from different strategies
        queries = []
        for _strat_name, strat_queries in strategies.items():
            queries.extend(strat_queries[:2])  # Take 2 from each strategy
        queries = queries[:max_queries]
    else:
        queries = strategies.get(strategy, strategies["broad"])[:max_queries]

    # Store generated queries
    search_session["queries"].extend(queries)

    # Store in memory
    search_memory.add_memory(
        content=f"Generated {len(queries)} search queries using {strategy} strategy for {base_topic}",
        memory_type=MemoryType.PROCEDURAL,
        agent_id="deepsearch_agent",
        tags=["query_generation", strategy, base_topic.lower()],
        importance_score=0.7,
    )

    result = f"🧠 Generated {len(queries)} search queries ({strategy} strategy):\n"
    for i, query in enumerate(queries, 1):
        result += f"{i:2d}. {query}\n"

    return result


def execute_search_batch(queries: list[str], max_results_per_query: int = 5) -> str:
    """Execute a batch of search queries and collect results."""
    start_time = datetime.now()
    all_results = []

    for _i, query in enumerate(queries, 1):
        try:
            # Use DuckDuckGo search
            search_result = GoogleSearch.static_call(query=query, max_results=max_results_per_query, region="us-en")

            if "results" in search_result:
                results = search_result["results"]
                all_results.extend(results)

                # Store each result
                for result in results:
                    search_memory.add_memory(
                        content=f"Search result: {result.get('title', '')} - {result.get('body', '')[:200]}",
                        memory_type=MemoryType.SEMANTIC,
                        agent_id="deepsearch_agent",
                        tags=["search_result", "web_search", query.split()[0].lower()],
                        importance_score=0.6,
                    )

        except Exception as e:
            print(f"Error searching for '{query}': {e}")

    processing_time = (datetime.now() - start_time).total_seconds()

    # Update session metrics
    search_session["performance_metrics"]["queries_executed"] += len(queries)
    search_session["performance_metrics"]["results_found"] += len(all_results)
    search_session["performance_metrics"]["processing_time"] += processing_time
    search_session["results"].extend(all_results)

    # Analyze result quality
    unique_domains = set()
    total_chars = 0

    for result in all_results:
        if "href" in result:
            domain = result["href"].split("/")[2] if len(result["href"].split("/")) > 2 else "unknown"
            unique_domains.add(domain)
        total_chars += len(result.get("body", "") + result.get("title", ""))

    summary = "🔍 Search Batch Complete:\n"
    summary += f"  • Queries executed: {len(queries)}\n"
    summary += f"  • Results found: {len(all_results)}\n"
    summary += f"  • Unique domains: {len(unique_domains)}\n"
    summary += f"  • Total content: {total_chars:,} characters\n"
    summary += f"  • Processing time: {processing_time:.2f}s\n"
    summary += f"  • Avg results/query: {len(all_results) / len(queries):.1f}\n"

    return summary


def analyze_search_results(min_results: int = 10) -> str:
    """Analyze collected search results for insights and patterns."""
    results = search_session["results"]

    if len(results) < min_results:
        return f"⚠️  Need at least {min_results} results for analysis (have {len(results)})"

    analysis = {
        "total_results": len(results),
        "domains": defaultdict(int),
        "keywords": defaultdict(int),
        "content_types": defaultdict(int),
        "quality_scores": [],
        "temporal_patterns": defaultdict(int),
    }

    # Analyze each result
    for result in results:
        # Domain analysis
        if "href" in result:
            try:
                domain = result["href"].split("/")[2]
                analysis["domains"][domain] += 1
            except (IndexError, KeyError):
                pass

        # Content analysis
        text_content = (result.get("title", "") + " " + result.get("body", "")).lower()

        # Extract keywords (simple frequency analysis)
        words = re.findall(r"\b\w{4,}\b", text_content)
        for word in words:
            if word not in ["this", "that", "with", "from", "they", "have", "been", "will", "more"]:
                analysis["keywords"][word] += 1

        # Quality scoring (simple heuristic)
        quality = 0
        if len(result.get("title", "")) > 10:
            quality += 1
        if len(result.get("body", "")) > 50:
            quality += 1
        if "href" in result and "wikipedia" in result["href"]:
            quality += 1
        if "href" in result and any(domain in result["href"] for domain in [".edu", ".gov", ".org"]):
            quality += 1

        analysis["quality_scores"].append(quality)

    # Generate insights
    top_domains = sorted(analysis["domains"].items(), key=lambda x: x[1], reverse=True)[:5]
    top_keywords = sorted(analysis["keywords"].items(), key=lambda x: x[1], reverse=True)[:10]
    avg_quality = sum(analysis["quality_scores"]) / len(analysis["quality_scores"]) if analysis["quality_scores"] else 0

    # Store analysis results
    search_session["insights"] = {
        "top_domains": top_domains,
        "top_keywords": top_keywords,
        "average_quality": avg_quality,
        "analysis_timestamp": datetime.now().isoformat(),
    }

    # Store in memory
    search_memory.add_memory(
        content=f"Analysis of {len(results)} search results: {len(top_domains)} domains, {len(top_keywords)} keywords",
        memory_type=MemoryType.SEMANTIC,
        agent_id="deepsearch_agent",
        tags=["analysis", "insights", "search_results"],
        importance_score=0.8,
    )

    # Format results
    report = "📊 Search Results Analysis:\n"
    report += f"{'=' * 40}\n"
    report += f"Total results analyzed: {analysis['total_results']}\n"
    report += f"Average quality score: {avg_quality:.1f}/4\n\n"

    report += "🏆 Top Domains:\n"
    for domain, count in top_domains:
        report += f"  • {domain}: {count} results\n"

    report += "\n🔤 Top Keywords:\n"
    for keyword, freq in top_keywords:
        report += f"  • {keyword}: {freq} mentions\n"

    # Quality distribution
    quality_dist = defaultdict(int)
    for score in analysis["quality_scores"]:
        quality_dist[score] += 1

    report += "\n📈 Quality Distribution:\n"
    for score in sorted(quality_dist.keys()):
        report += f"  • Score {score}: {quality_dist[score]} results\n"

    return report


def extract_entities_from_results() -> str:
    """Extract and analyze entities from search results."""
    all_content = ""

    # Combine all result content
    for result in search_session["results"]:
        content = result.get("title", "") + " " + result.get("body", "")
        all_content += content + " "

    if len(all_content) < 100:
        return "⚠️  Not enough content for entity extraction"

    # Extract entities using the EntityExtractor tool
    entity_result = EntityExtractor.static_call(
        text=all_content[:5000],  # Limit content size
        entity_types=["emails", "urls", "phone_numbers", "dates", "numbers", "names"],
    )

    entities = entity_result.get("entities", {})
    total_entities = entity_result.get("total_entities", 0)

    # Store entity extraction results
    search_memory.add_memory(
        content=f"Extracted {total_entities} entities from search results",
        memory_type=MemoryType.SEMANTIC,
        agent_id="deepsearch_agent",
        tags=["entity_extraction", "analysis"],
        importance_score=0.7,
    )

    # Format results
    report = "🏷️  Entity Extraction Results:\n"
    report += f"{'=' * 40}\n"
    report += f"Total entities found: {total_entities}\n\n"

    for entity_type, entity_list in entities.items():
        if entity_list:
            report += f"📋 {entity_type.title()}:\n"
            for entity in entity_list[:10]:  # Show max 10 per type
                report += f"  • {entity}\n"
            if len(entity_list) > 10:
                report += f"  ... and {len(entity_list) - 10} more\n"
            report += "\n"

    return report


def classify_content_sentiment() -> str:
    """Classify the sentiment and topics of search results."""
    sentiments = {"positive": 0, "negative": 0, "neutral": 0}
    topics = defaultdict(int)

    # Analyze each result
    for result in search_session["results"]:
        content = result.get("title", "") + " " + result.get("body", "")

        if len(content) > 20:
            # Sentiment analysis
            sentiment_result = TextClassifier.static_call(
                text=content[:1000],  # Limit content
                method="sentiment",
            )

            sentiment = sentiment_result.get("sentiment", "neutral")
            sentiments[sentiment] += 1

            # Topic classification
            topic_result = TextClassifier.static_call(text=content[:1000], method="topic")

            topic = topic_result.get("topic", "general")
            topics[topic] += 1

    total_classified = sum(sentiments.values())

    # Store classification results
    search_memory.add_memory(
        content=f"Classified {total_classified} search results: {dict(sentiments)} sentiment distribution",
        memory_type=MemoryType.SEMANTIC,
        agent_id="deepsearch_agent",
        tags=["classification", "sentiment", "topic"],
        importance_score=0.6,
    )

    # Format results
    report = "🎭 Content Classification:\n"
    report += f"{'=' * 40}\n"
    report += f"Results classified: {total_classified}\n\n"

    # Sentiment distribution
    report += "😊 Sentiment Analysis:\n"
    for sentiment, count in sentiments.items():
        percentage = (count / total_classified * 100) if total_classified > 0 else 0
        report += f"  • {sentiment.title()}: {count} ({percentage:.1f}%)\n"

    # Topic distribution
    top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
    report += "\n📚 Top Topics:\n"
    for topic, count in top_topics:
        percentage = (count / total_classified * 100) if total_classified > 0 else 0
        report += f"  • {topic.title()}: {count} ({percentage:.1f}%)\n"

    return report


def generate_content_summary(max_length: int = 500) -> str:
    """Generate a summary of all collected content."""
    # Combine content from top results
    content_pieces = []

    for result in search_session["results"][:20]:  # Use top 20 results
        title = result.get("title", "")
        body = result.get("body", "")

        if title and body:
            piece = f"{title}. {body}"
            content_pieces.append(piece)

    combined_content = " ".join(content_pieces)

    if len(combined_content) < 100:
        return "⚠️  Not enough content for summarization"

    # Generate summary using TextSummarizer
    summary_result = TextSummarizer.static_call(
        text=combined_content[:5000],  # Limit input
        method="extractive",
        max_sentences=5,
        max_length=max_length,
    )

    summary = summary_result.get("summary", "Could not generate summary")
    original_length = summary_result.get("original_length", 0)
    summary_length = summary_result.get("summary_length", 0)
    compression_ratio = summary_result.get("compression_ratio", 0)

    # Store summary
    search_memory.add_memory(
        content=f"Generated content summary: {summary[:100]}...",
        memory_type=MemoryType.SEMANTIC,
        agent_id="deepsearch_agent",
        tags=["summary", "content"],
        importance_score=0.9,
    )

    # Format results
    report = "📝 Content Summary:\n"
    report += f"{'=' * 40}\n"
    report += f"Original: {original_length:,} chars → Summary: {summary_length:,} chars\n"
    report += f"Compression ratio: {compression_ratio:.1%}\n\n"
    report += f"📋 Summary:\n{summary}\n"

    return report


def build_knowledge_graph() -> str:
    """Build a knowledge graph from search insights."""
    insights = search_session["insights"]

    if not insights:
        return "⚠️  No insights available for knowledge graph"

    # Create knowledge graph structure
    graph = {
        "nodes": [],
        "connections": [],
        "metadata": {
            "created": datetime.now().isoformat(),
            "topic": search_session.get("topic", "unknown"),
            "session_id": search_session.get("session_id", "unknown"),
        },
    }

    # Add topic as central node
    central_topic = search_session.get("topic", "Research Topic")
    graph["nodes"].append({"id": "central", "label": central_topic, "type": "topic", "importance": 1.0})

    # Add keyword nodes
    for keyword, freq in insights.get("top_keywords", [])[:10]:
        graph["nodes"].append(
            {
                "id": f"keyword_{keyword}",
                "label": keyword,
                "type": "keyword",
                "importance": min(freq / 10, 1.0),
                "frequency": freq,
            }
        )

        # Connect to central topic
        graph["connections"].append(
            {"from": "central", "to": f"keyword_{keyword}", "type": "contains", "strength": min(freq / 10, 1.0)}
        )

    # Add domain nodes
    for domain, count in insights.get("top_domains", [])[:5]:
        graph["nodes"].append(
            {
                "id": f"domain_{domain}",
                "label": domain,
                "type": "source",
                "importance": min(count / 5, 1.0),
                "result_count": count,
            }
        )

        # Connect to central topic
        graph["connections"].append(
            {"from": "central", "to": f"domain_{domain}", "type": "sources_from", "strength": min(count / 5, 1.0)}
        )

    # Store in session
    search_session["knowledge_graph"] = graph

    # Store in memory
    search_memory.add_memory(
        content=f"Built knowledge graph with {len(graph['nodes'])} nodes and {len(graph['connections'])} connections",
        memory_type=MemoryType.SEMANTIC,
        agent_id="deepsearch_agent",
        tags=["knowledge_graph", "visualization"],
        importance_score=0.8,
    )

    # Format results
    report = "🕸️  Knowledge Graph:\n"
    report += f"{'=' * 40}\n"
    report += f"Central topic: {central_topic}\n"
    report += f"Total nodes: {len(graph['nodes'])}\n"
    report += f"Total connections: {len(graph['connections'])}\n\n"

    report += "🔍 Key Concepts:\n"
    for node in graph["nodes"][1:6]:  # Skip central, show top 5
        if node["type"] == "keyword":
            report += f"  • {node['label']} (mentioned {node['frequency']} times)\n"

    report += "\n📡 Top Sources:\n"
    for node in graph["nodes"]:
        if node["type"] == "source":
            report += f"  • {node['label']} ({node['result_count']} results)\n"

    return report


def generate_research_report() -> str:
    """Generate a comprehensive research report."""
    session = search_session
    start_time = session.get("start_time", datetime.now())
    duration = datetime.now() - start_time

    report = f"""
🔍 DEEPSEARCH RESEARCH REPORT
{"=" * 60}

📊 EXECUTIVE SUMMARY
Topic: {session.get("topic", "Unknown")}
Session ID: {session.get("session_id", "Unknown")}
Duration: {duration.total_seconds():.0f} seconds
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

📈 SEARCH PERFORMANCE
• Queries executed: {session["performance_metrics"]["queries_executed"]}
• Results collected: {session["performance_metrics"]["results_found"]}
• Processing time: {session["performance_metrics"]["processing_time"]:.2f}s
• Avg results/query: {
        (
            session["performance_metrics"]["results_found"] / max(session["performance_metrics"]["queries_executed"], 1)
        ):.1f}

🎯 KEY FINDINGS
"""

    # Add insights if available
    insights = session.get("insights", {})
    if insights:
        report += f"• {len(insights.get('top_keywords', []))} key concepts identified\n"
        report += f"• {len(insights.get('top_domains', []))} primary sources analyzed\n"
        report += f"• Content quality score: {insights.get('average_quality', 0):.1f}/4\n"

        if insights.get("top_keywords"):
            report += "\n🔤 TOP CONCEPTS:\n"
            for keyword, freq in insights["top_keywords"][:5]:
                report += f"• {keyword} ({freq} mentions)\n"

        if insights.get("top_domains"):
            report += "\n📡 PRIMARY SOURCES:\n"
            for domain, count in insights["top_domains"][:5]:
                report += f"• {domain} ({count} results)\n"

    # Add memory statistics
    if search_memory:
        stats = search_memory.get_statistics()
        report += "\n🧠 KNOWLEDGE BASE:\n"
        report += f"• Total memories: {stats['total_memories']}\n"
        report += f"• Memory types: {', '.join(stats['by_type'].keys())}\n"

        # Get important memories
        important_memories = search_memory.retrieve_memories(
            tags=[session.get("topic", "").lower()], min_importance=0.7, limit=3
        )

        if important_memories:
            report += "\n⭐ KEY INSIGHTS:\n"
            for i, memory in enumerate(important_memories, 1):
                report += f"{i}. {memory.content[:100]}...\n"

    # Add knowledge graph info
    graph = session.get("knowledge_graph", {})
    if graph:
        report += "\n🕸️  KNOWLEDGE GRAPH:\n"
        report += f"• Nodes: {len(graph.get('nodes', []))}\n"
        report += f"• Connections: {len(graph.get('connections', []))}\n"

    # Add recommendations
    report += "\n💡 RECOMMENDATIONS:\n"

    if session["performance_metrics"]["results_found"] < 50:
        report += "• Consider expanding search with additional query strategies\n"

    if insights.get("average_quality", 0) < 2:
        report += "• Focus on higher-quality sources (.edu, .gov, .org domains)\n"

    report += "• Follow up with specific queries on top concepts\n"
    report += "• Consider academic database searches for deeper research\n"

    report += f"\n{'=' * 60}\n"
    report += "End of Report\n"

    return report


def save_search_session(include_raw_data: bool = True) -> str:
    """Save the complete search session to files."""
    session_dir = Path.home() / ".xerxes" / "deepsearch_sessions" / search_session.get("session_id", "unknown")
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save summary report
    report = generate_research_report()
    report_path = session_dir / "research_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save session data as JSON
    session_data = {
        "metadata": {
            "topic": search_session.get("topic"),
            "session_id": search_session.get("session_id"),
            "start_time": search_session.get("start_time").isoformat() if search_session.get("start_time") else None,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - search_session.get("start_time", datetime.now())).total_seconds(),
        },
        "performance": search_session["performance_metrics"],
        "insights": search_session.get("insights", {}),
        "knowledge_graph": search_session.get("knowledge_graph", {}),
        "queries": search_session.get("queries", []),
    }

    if include_raw_data:
        session_data["raw_results"] = search_session.get("results", [])

    json_path = session_dir / "session_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, default=str)

    # Save knowledge graph in a format suitable for visualization
    if search_session.get("knowledge_graph"):
        graph_path = session_dir / "knowledge_graph.json"
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(search_session["knowledge_graph"], f, indent=2)

    # Store save operation in memory
    search_memory.add_memory(
        content=f"Search session saved to {session_dir}",
        memory_type=MemoryType.EPISODIC,
        agent_id="deepsearch_agent",
        tags=["save", "export"],
        importance_score=0.6,
    )

    result = "💾 Search Session Saved:\n"
    result += f"📁 Directory: {session_dir}\n"
    result += "📄 Files created:\n"
    result += f"  • research_report.txt ({report_path.stat().st_size:,} bytes)\n"
    result += f"  • session_data.json ({json_path.stat().st_size:,} bytes)\n"

    if search_session.get("knowledge_graph"):
        result += f"  • knowledge_graph.json ({graph_path.stat().st_size:,} bytes)\n"

    return result


async def main():
    """Run the DeepSearch Agent demonstration."""
    print("=" * 80)
    print("🔍 DEEPSEARCH AGENT - ADVANCED INFORMATION DISCOVERY DEMO")
    print("=" * 80)
    print()

    # Create DeepSearch agent with comprehensive tools
    agent = Agent(
        id="deepsearch_agent",
        name="DeepSearch Intelligence Agent",
        model="gpt-4o",
        instructions="""You are DeepSearch, an advanced information discovery and analysis agent.

        Your capabilities include:
        1. Multi-strategy search execution and optimization
        2. Intelligent content analysis and synthesis
        3. Entity extraction and pattern recognition
        4. Knowledge graph construction
        5. Comprehensive research reporting

        Your mission is to conduct thorough, intelligent searches and provide
        comprehensive analysis of information across multiple sources and perspectives.

        Use your tools strategically to build deep understanding of topics through:
        - Multi-layered search strategies
        - Content quality assessment
        - Pattern and entity recognition
        - Knowledge synthesis and visualization

        Always strive for accuracy, comprehensiveness, and actionable insights.""",
        functions=[
            # Core search and analysis functions
            initialize_search_session,
            generate_search_queries,
            execute_search_batch,
            analyze_search_results,
            extract_entities_from_results,
            classify_content_sentiment,
            generate_content_summary,
            build_knowledge_graph,
            generate_research_report,
            save_search_session,
            # Xerxes tools for enhanced capabilities
            GoogleSearch,
            WebScraper,
            URLAnalyzer,
            EntityExtractor,
            TextClassifier,
            TextSummarizer,
            TextSimilarity,
            TextProcessor,
            JSONProcessor,
            DataConverter,
            StatisticalAnalyzer,
            WriteFile,
            ReadFile,
        ],
        max_tokens=1500,
        temperature=0.3,  # Lower temperature for more focused analysis
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    # Initialize Xerxes with enhanced memory
    xerxes = Xerxes(client, enable_memory=True)
    xerxes.memory = search_memory
    xerxes.register_agent(agent)

    # Demo topics for comprehensive search
    demo_topics = [
        "artificial intelligence safety research",
        "quantum computing applications",
        "sustainable energy technologies",
        "biomedical engineering innovations",
    ]

    print("🎯 Demo Topics Available:")
    for i, topic in enumerate(demo_topics, 1):
        print(f"  {i}. {topic}")

    # For demo, use the first topic
    selected_topic = demo_topics[0]
    print(f"\n🔍 Starting DeepSearch for: '{selected_topic}'\n")

    messages = MessagesHistory(messages=[])

    # Phase 1: Initialize Search Session
    print("=" * 60)
    print("PHASE 1: SEARCH INITIALIZATION")
    print("=" * 60)

    init_result = initialize_search_session(selected_topic)
    print(init_result)

    # Phase 2: Generate Search Queries
    print("\n" + "=" * 60)
    print("PHASE 2: INTELLIGENT QUERY GENERATION")
    print("=" * 60)

    query_result = generate_search_queries(selected_topic, "comprehensive", 8)
    print(query_result)

    # Phase 3: Execute Search Batch
    print("\n" + "=" * 60)
    print("PHASE 3: SEARCH EXECUTION")
    print("=" * 60)

    queries_to_execute = search_session["queries"][:6]  # Execute first 6 queries
    search_result = execute_search_batch(queries_to_execute, 4)
    print(search_result)

    # Phase 4: Content Analysis
    print("\n" + "=" * 60)
    print("PHASE 4: CONTENT ANALYSIS")
    print("=" * 60)

    analysis_result = analyze_search_results(5)
    print(analysis_result)

    # Phase 5: Entity Extraction
    print("\n" + "=" * 60)
    print("PHASE 5: ENTITY EXTRACTION")
    print("=" * 60)

    entity_result = extract_entities_from_results()
    print(entity_result)

    # Phase 6: Content Classification
    print("\n" + "=" * 60)
    print("PHASE 6: SENTIMENT & TOPIC CLASSIFICATION")
    print("=" * 60)

    classification_result = classify_content_sentiment()
    print(classification_result)

    # Phase 7: Content Summarization
    print("\n" + "=" * 60)
    print("PHASE 7: CONTENT SUMMARIZATION")
    print("=" * 60)

    summary_result = generate_content_summary(400)
    print(summary_result)

    # Phase 8: Knowledge Graph Construction
    print("\n" + "=" * 60)
    print("PHASE 8: KNOWLEDGE GRAPH BUILDING")
    print("=" * 60)

    graph_result = build_knowledge_graph()
    print(graph_result)

    # Phase 9: AI Agent Interaction
    print("\n" + "=" * 60)
    print("PHASE 9: AI AGENT SYNTHESIS")
    print("=" * 60)

    # Ask the agent to synthesize findings
    synthesis_query = (
        f"Based on our comprehensive search and analysis of '{selected_topic}', "
        "provide a strategic synthesis of the key findings, emerging patterns, and actionable insights. "
        "Include your assessment of the research quality and recommendations for further investigation."
    )

    messages.messages.append(UserMessage(content=synthesis_query))

    print(f"🤖 Agent Query: {synthesis_query[:100]}...\n")
    print("🔄 Generating AI synthesis...\n")

    try:
        response = await xerxes.create_response(
            prompt=synthesis_query,
            messages=messages,
            agent_id=agent.id,
            stream=False,
            apply_functions=True,
        )

        print("🧠 DeepSearch Agent Synthesis:")
        print("-" * 40)
        print(response)
        print("-" * 40)

    except Exception as e:
        print(f"❌ Error in agent synthesis: {e}")

    # Phase 10: Generate Final Report
    print("\n" + "=" * 60)
    print("PHASE 10: RESEARCH REPORT GENERATION")
    print("=" * 60)

    final_report = generate_research_report()
    print(final_report[:1000] + "..." if len(final_report) > 1000 else final_report)

    # Phase 11: Save Session
    print("\n" + "=" * 60)
    print("PHASE 11: SESSION PERSISTENCE")
    print("=" * 60)

    save_result = save_search_session(include_raw_data=True)
    print(save_result)

    # Final Statistics
    print("\n" + "=" * 60)
    print("📊 FINAL STATISTICS")
    print("=" * 60)

    if search_memory:
        stats = search_memory.get_statistics()
        print("\n🧠 Memory Statistics:")
        print(f"  • Total memories: {stats['total_memories']}")
        print(f"  • Memory distribution: {stats['by_type']}")
        print(f"  • Cache hit rate: {stats['cache_hit_rate']:.1%}")

    session_duration = datetime.now() - search_session.get("start_time", datetime.now())
    print("\n⏱️  Session Performance:")
    print(f"  • Total duration: {session_duration.total_seconds():.1f}s")
    print(f"  • Queries executed: {search_session['performance_metrics']['queries_executed']}")
    print(f"  • Results collected: {search_session['performance_metrics']['results_found']}")
    processing_efficiency = (
        search_session["performance_metrics"]["results_found"] / search_session["performance_metrics"]["processing_time"]
    )
    print(f"  • Processing efficiency: {processing_efficiency:.1f} results/second")

    # Knowledge graph statistics
    graph = search_session.get("knowledge_graph", {})
    if graph:
        print("\n🕸️  Knowledge Graph:")
        print(f"  • Nodes created: {len(graph.get('nodes', []))}")
        print(f"  • Connections mapped: {len(graph.get('connections', []))}")
        print(f"  • Graph density: {len(graph.get('connections', [])) / max(len(graph.get('nodes', [])), 1):.2f}")

    # Save memory
    search_memory.save()
    print("\n💾 Memory state persisted")

    print("\n" + "=" * 80)
    print("✅ DEEPSEARCH AGENT DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    print("\n📋 Summary:")
    print(f"  • Conducted comprehensive research on '{selected_topic}'")
    print(f"  • Executed multi-strategy search with {search_session['performance_metrics']['queries_executed']} queries")
    top_domains_count = len(search_session.get("insights", {}).get("top_domains", []))
    results_found = search_session["performance_metrics"]["results_found"]
    print(f"  • Analyzed {results_found} results from {top_domains_count} sources")
    print(f"  • Built knowledge graph with {len(graph.get('nodes', []))} concepts")
    print("  • Generated comprehensive research report")
    print("  • Demonstrated advanced AI-powered information discovery")


if __name__ == "__main__":
    asyncio.run(main())
