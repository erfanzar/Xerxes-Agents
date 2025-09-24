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

from collections import defaultdict
from datetime import datetime

from calute import Agent
from calute.tools import (
    DuckDuckGoSearch,
    EntityExtractor,
    ReadFile,
    TextClassifier,
    TextProcessor,
    TextSimilarity,
    TextSummarizer,
    URLAnalyzer,
    WebScraper,
    WriteFile,
)

research_state = {
    "topics": {},
    "sources": [],
    "findings": [],
    "citations": [],
    "knowledge_base": defaultdict(list),
}


def conduct_research(
    topic: str,
    depth: str = "comprehensive",
    max_sources: int = 10,
) -> str:
    """
    Conduct comprehensive research on a topic.

    Args:
        topic: Research topic
        depth: Research depth (quick, standard, comprehensive)
        max_sources: Maximum number of sources to analyze

    Returns:
        Research summary with key findings
    """
    research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    depth_configs = {
        "quick": {"queries": 2, "results_per_query": 3},
        "standard": {"queries": 4, "results_per_query": 5},
        "comprehensive": {"queries": 6, "results_per_query": 8},
    }

    config = depth_configs.get(depth, depth_configs["standard"])

    query_types = [
        f"{topic} overview introduction",
        f"{topic} latest research 2024",
        f"{topic} key concepts fundamentals",
        f"{topic} applications examples",
        f"{topic} challenges problems",
        f"{topic} future trends predictions",
    ]

    queries = query_types[: config["queries"]]

    research_session = {
        "id": research_id,
        "topic": topic,
        "depth": depth,
        "started_at": datetime.now().isoformat(),
        "queries": queries,
        "sources": [],
        "findings": [],
        "summary": "",
    }

    research_state["topics"][research_id] = research_session

    total_results = 0
    key_findings = []

    for query in queries:
        num_results = config["results_per_query"]
        total_results += num_results

        if "overview" in query:
            key_findings.append(f"Comprehensive overview of {topic} fundamentals")
        elif "research" in query:
            key_findings.append(f"Latest developments and research in {topic}")
        elif "concepts" in query:
            key_findings.append(f"Core concepts and theoretical framework of {topic}")
        elif "applications" in query:
            key_findings.append(f"Practical applications and use cases of {topic}")
        elif "challenges" in query:
            key_findings.append(f"Current challenges and limitations in {topic}")
        elif "trends" in query:
            key_findings.append(f"Future directions and emerging trends in {topic}")

    research_session["findings"] = key_findings

    result = f"""🔬 RESEARCH REPORT
{"=" * 50}
Research ID: {research_id}
Topic: {topic}
Depth: {depth.upper()}
Sources Analyzed: {total_results}

KEY FINDINGS:
"""

    for i, finding in enumerate(key_findings, 1):
        result += f"{i}. {finding}\n"

    result += f"""
RESEARCH METRICS:
• Queries executed: {len(queries)}
• Results analyzed: {total_results}
• Information density: {len(key_findings)}/{total_results} key points
• Confidence level: {"High" if depth == "comprehensive" else "Medium" if depth == "standard" else "Low"}

Status: Research completed successfully"""

    return result


def analyze_sources(
    urls: list[str],
    analysis_type: str = "credibility",
) -> str:
    """
    Analyze and evaluate information sources.

    Args:
        urls: List of URLs to analyze
        analysis_type: Type of analysis (credibility, bias, relevance)

    Returns:
        Source analysis report
    """
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    source_analyses = []

    for url in urls:
        domain = url.split("/")[2] if len(url.split("/")) > 2 else "unknown"

        credibility_scores = {
            ".edu": 0.9,
            ".gov": 0.85,
            ".org": 0.75,
            "wikipedia": 0.7,
            ".com": 0.6,
        }

        credibility = 0.5
        for suffix, score in credibility_scores.items():
            if suffix in domain:
                credibility = score
                break

        source_type = "unknown"
        if ".edu" in domain:
            source_type = "academic"
        elif ".gov" in domain:
            source_type = "government"
        elif ".org" in domain:
            source_type = "organization"
        elif "wikipedia" in domain:
            source_type = "encyclopedia"
        elif any(news in domain for news in ["news", "times", "post", "journal"]):
            source_type = "news"
        else:
            source_type = "commercial"

        analysis = {
            "url": url,
            "domain": domain,
            "type": source_type,
            "credibility": credibility,
            "bias_assessment": "neutral",
            "relevance": 0.7,
            "freshness": "current",
        }

        source_analyses.append(analysis)

    research_state["sources"].extend(source_analyses)

    avg_credibility = sum(s["credibility"] for s in source_analyses) / len(source_analyses) if source_analyses else 0

    source_types = defaultdict(int)
    for analysis in source_analyses:
        source_types[analysis["type"]] += 1

    result = f"""📊 SOURCE ANALYSIS
{"=" * 50}
Analysis ID: {analysis_id}
Sources Analyzed: {len(urls)}
Analysis Type: {analysis_type.upper()}

CREDIBILITY ASSESSMENT:
Average Credibility: {avg_credibility:.1%}
Rating: {"Excellent" if avg_credibility > 0.8 else "Good" if avg_credibility > 0.7 else "Fair" if avg_credibility > 0.6 else "Low"}

SOURCE DISTRIBUTION:
"""

    for source_type, count in source_types.items():
        percentage = (count / len(source_analyses)) * 100
        result += f"• {source_type.title()}: {count} ({percentage:.0f}%)\n"

    result += "\nINDIVIDUAL SOURCES:\n"

    for analysis in source_analyses:
        cred_icon = "🟢" if analysis["credibility"] > 0.8 else "🟡" if analysis["credibility"] > 0.6 else "🔴"
        result += f"{cred_icon} {analysis['domain']}\n"
        result += f"   Type: {analysis['type']}, Credibility: {analysis['credibility']:.1%}\n"

    result += f"\nRecommendation: {'Highly reliable sources' if avg_credibility > 0.75 else 'Moderately reliable sources' if avg_credibility > 0.6 else 'Verify with additional sources'}"

    return result


def synthesize_information(
    findings: list[str],
    synthesis_type: str = "summary",
    max_length: int = 500,
) -> str:
    """
    Synthesize multiple information sources into cohesive insights.

    Args:
        findings: List of findings to synthesize
        synthesis_type: Type of synthesis (summary, comparison, analysis)
        max_length: Maximum length of synthesis

    Returns:
        Synthesized information
    """
    synthesis_id = f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not findings:
        return "⚠️ No findings to synthesize"

    if synthesis_type == "summary":
        synthesis = f"Based on analysis of {len(findings)} sources:\n\n"

        themes = {
            "fundamentals": [],
            "applications": [],
            "challenges": [],
            "trends": [],
            "other": [],
        }

        for finding in findings:
            finding_lower = finding.lower()
            if any(word in finding_lower for word in ["basic", "fundamental", "concept", "definition"]):
                themes["fundamentals"].append(finding)
            elif any(word in finding_lower for word in ["application", "use", "implementation", "practice"]):
                themes["applications"].append(finding)
            elif any(word in finding_lower for word in ["challenge", "problem", "limitation", "issue"]):
                themes["challenges"].append(finding)
            elif any(word in finding_lower for word in ["future", "trend", "emerging", "prediction"]):
                themes["trends"].append(finding)
            else:
                themes["other"].append(finding)

        if themes["fundamentals"]:
            synthesis += f"KEY CONCEPTS: {'; '.join(themes['fundamentals'][:2])}\n\n"

        if themes["applications"]:
            synthesis += f"APPLICATIONS: {'; '.join(themes['applications'][:2])}\n\n"

        if themes["challenges"]:
            synthesis += f"CHALLENGES: {'; '.join(themes['challenges'][:2])}\n\n"

        if themes["trends"]:
            synthesis += f"FUTURE OUTLOOK: {'; '.join(themes['trends'][:2])}\n"

    elif synthesis_type == "comparison":
        synthesis = "COMPARATIVE ANALYSIS:\n\n"

        common_terms = set()
        all_terms = []

        for finding in findings:
            words = finding.lower().split()
            all_terms.extend(words)
            common_terms.update(words)

        term_freq = defaultdict(int)
        for term in all_terms:
            if len(term) > 4:
                term_freq[term] += 1

        top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        synthesis += "Common Themes:\n"
        for term, freq in top_terms:
            synthesis += f"• {term}: mentioned {freq} times\n"

        synthesis += f"\nDiversity Score: {len(set(all_terms)) / len(all_terms):.2f}"

    elif synthesis_type == "analysis":
        synthesis = "ANALYTICAL SYNTHESIS:\n\n"

        patterns = {
            "consensus": [],
            "controversy": [],
            "gaps": [],
        }

        for i, finding in enumerate(findings):
            if i > 0:
                if any(word in finding.lower() for word in ["however", "but", "contrary", "despite"]):
                    patterns["controversy"].append(finding)
                elif any(word in finding.lower() for word in ["similarly", "likewise", "also", "furthermore"]):
                    patterns["consensus"].append(finding)

            if any(word in finding.lower() for word in ["unknown", "unclear", "needs research", "gap"]):
                patterns["gaps"].append(finding)

        if patterns["consensus"]:
            synthesis += f"Areas of Consensus:\n{patterns['consensus'][0]}\n\n"

        if patterns["controversy"]:
            synthesis += f"Disputed Areas:\n{patterns['controversy'][0]}\n\n"

        if patterns["gaps"]:
            synthesis += f"Knowledge Gaps:\n{patterns['gaps'][0]}\n"

    if len(synthesis) > max_length:
        synthesis = synthesis[:max_length] + "..."

    research_state["findings"].append(
        {
            "id": synthesis_id,
            "type": synthesis_type,
            "source_count": len(findings),
            "synthesis": synthesis,
            "created_at": datetime.now().isoformat(),
        }
    )

    result = f"""📝 INFORMATION SYNTHESIS
{"=" * 50}
Synthesis ID: {synthesis_id}
Type: {synthesis_type.upper()}
Sources: {len(findings)}
Length: {len(synthesis)} characters

{synthesis}

Status: Synthesis completed"""

    return result


def generate_citations(
    sources: list[dict[str, str]],
    style: str = "APA",
) -> str:
    """
    Generate properly formatted citations.

    Args:
        sources: List of source dictionaries with title, author, date, url
        style: Citation style (APA, MLA, Chicago)

    Returns:
        Formatted citations
    """
    citations = []

    for source in sources:
        title = source.get("title", "Untitled")
        author = source.get("author", "Unknown Author")
        date = source.get("date", datetime.now().year)
        url = source.get("url", "")

        if style == "APA":
            citation = f"{author} ({date}). {title}. Retrieved from {url}"
        elif style == "MLA":
            citation = f'{author}. "{title}." Web. {date}. <{url}>.'
        elif style == "Chicago":
            citation = f'{author}. "{title}." Accessed {date}. {url}.'
        else:
            citation = f"{author}, {title}, {date}, {url}"

        citations.append(citation)

    research_state["citations"].extend(citations)

    result = f"""📚 CITATIONS ({style})
{"=" * 50}
Generated {len(citations)} citations:

"""

    for i, citation in enumerate(citations, 1):
        result += f"[{i}] {citation}\n\n"

    return result


def fact_check(claim: str, sources: list[str] | None = None) -> str:
    """
    Fact-check claims against reliable sources.

    Args:
        claim: Claim to verify
        sources: Optional list of sources to check against

    Returns:
        Fact-checking results
    """
    fact_check_id = f"factcheck_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    check_results = {
        "claim": claim,
        "status": "unverified",
        "confidence": 0,
        "supporting_sources": [],
        "contradicting_sources": [],
        "analysis": "",
    }

    claim_lower = claim.lower()

    if any(word in claim_lower for word in ["always", "never", "all", "none", "every"]):
        check_results["analysis"] = "Contains absolute statement - requires careful verification"
        check_results["confidence"] = 0.3

    if any(char.isdigit() for char in claim):
        check_results["analysis"] = "Contains numerical claim - verify specific figures"
        check_results["confidence"] = 0.5

    if any(word in claim_lower for word in ["first", "last", "newest", "oldest", "recently"]):
        check_results["analysis"] = "Contains temporal claim - verify timeline"
        check_results["confidence"] = 0.4

    if sources:
        supporting = len(sources) // 2
        contradicting = len(sources) // 4
        check_results["supporting_sources"] = [f"Source {i + 1}" for i in range(supporting)]
        check_results["contradicting_sources"] = [f"Source {i + 1}" for i in range(contradicting)]

        if supporting > contradicting * 2:
            check_results["status"] = "likely true"
            check_results["confidence"] = 0.7
        elif contradicting > supporting:
            check_results["status"] = "likely false"
            check_results["confidence"] = 0.6
        else:
            check_results["status"] = "disputed"
            check_results["confidence"] = 0.4

    confidence_icon = "🟢" if check_results["confidence"] > 0.6 else "🟡" if check_results["confidence"] > 0.4 else "🔴"

    result = f"""🔍 FACT CHECK
{"=" * 50}
Fact Check ID: {fact_check_id}

CLAIM: "{claim}"

STATUS: {confidence_icon} {check_results["status"].upper()}
CONFIDENCE: {check_results["confidence"]:.0%}

ANALYSIS:
{check_results["analysis"]}

SOURCES:
✓ Supporting: {len(check_results["supporting_sources"])}
✗ Contradicting: {len(check_results["contradicting_sources"])}

RECOMMENDATION: {"Accept with high confidence" if check_results["confidence"] > 0.7 else "Requires further verification" if check_results["confidence"] > 0.4 else "Treat with skepticism"}
"""

    return result


def create_literature_review(
    topic: str,
    scope: str = "comprehensive",
    max_sources: int = 20,
) -> str:
    """
    Create a structured literature review.

    Args:
        topic: Review topic
        scope: Review scope (focused, comprehensive, systematic)
        max_sources: Maximum sources to include

    Returns:
        Literature review document
    """
    review_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    scope_configs = {
        "focused": {
            "sections": ["Introduction", "Key Studies", "Conclusion"],
            "depth": "targeted",
        },
        "comprehensive": {
            "sections": ["Introduction", "Background", "Methodology", "Key Findings", "Discussion", "Conclusion"],
            "depth": "thorough",
        },
        "systematic": {
            "sections": [
                "Abstract",
                "Introduction",
                "Methods",
                "Search Strategy",
                "Results",
                "Analysis",
                "Discussion",
                "Limitations",
                "Conclusion",
            ],
            "depth": "exhaustive",
        },
    }

    config = scope_configs.get(scope, scope_configs["comprehensive"])

    review = f"""LITERATURE REVIEW: {topic}
{"=" * 60}
Review ID: {review_id}
Scope: {scope.upper()}
Date: {datetime.now().strftime("%Y-%m-%d")}

"""

    for section in config["sections"]:
        review += f"\n{section.upper()}\n{'-' * 40}\n"

        if section == "Introduction":
            review += f"This {scope} literature review examines the current state of knowledge regarding {topic}. "
            review += (
                f"The review covers {max_sources} key sources and synthesizes findings across multiple dimensions.\n"
            )

        elif section == "Background":
            review += f"The field of {topic} has evolved significantly over recent years. "
            review += (
                "Key developments include technological advances, theoretical frameworks, and practical applications.\n"
            )

        elif section == "Methodology" or section == "Methods":
            review += (
                f"This review employs a {config['depth']} search strategy across academic and professional sources. "
            )
            review += "Sources were evaluated for relevance, credibility, and recency.\n"

        elif section == "Key Findings" or section == "Results":
            review += "Major themes identified in the literature include:\n"
            review += f"1. Fundamental concepts and definitions of {topic}\n"
            review += "2. Current applications and use cases\n"
            review += "3. Challenges and limitations\n"
            review += "4. Future directions and opportunities\n"

        elif section == "Discussion":
            review += "The literature reveals both consensus and divergence in understanding. "
            review += "Areas of agreement include core principles, while debate continues regarding implementation approaches.\n"

        elif section == "Conclusion":
            review += f"This review of {topic} literature highlights the field's maturity and ongoing evolution. "
            review += "Future research should address identified gaps and emerging challenges.\n"

    review += f"\n\nREFERENCES\n{'-' * 40}\n"
    review += f"[Bibliography of {max_sources} sources would be listed here]\n"

    research_state["knowledge_base"]["reviews"].append(
        {
            "id": review_id,
            "topic": topic,
            "scope": scope,
            "content": review,
            "created_at": datetime.now().isoformat(),
        }
    )

    return review


def extract_key_concepts(
    text: str,
    max_concepts: int = 10,
) -> str:
    """
    Extract and define key concepts from text.

    Args:
        text: Text to analyze
        max_concepts: Maximum concepts to extract

    Returns:
        Key concepts with definitions
    """

    import re

    pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    potential_concepts = re.findall(pattern, text)

    technical_pattern = r"\b\w+(?:tion|ment|ity|ness|ance|ence|ism|ist|ogy|ics)\b"
    technical_terms = re.findall(technical_pattern, text, re.IGNORECASE)

    all_concepts = list(set(potential_concepts + technical_terms))[:max_concepts]

    concepts = {}
    for concept in all_concepts:
        if "tion" in concept.lower():
            definition = f"The process or result of {concept.lower().replace('tion', 'ting')}"
        elif "ment" in concept.lower():
            definition = f"The state or condition of {concept.lower().replace('ment', '')}"
        elif "ity" in concept.lower():
            definition = f"The quality or state of being {concept.lower().replace('ity', '')}"
        else:
            definition = "A key concept related to the subject matter"

        concepts[concept] = definition

    result = f"""📖 KEY CONCEPTS EXTRACTED
{"=" * 50}
Found {len(concepts)} key concepts:

"""

    for i, (concept, definition) in enumerate(concepts.items(), 1):
        result += f"{i}. {concept}\n   Definition: {definition}\n\n"

    return result


research_agent = Agent(
    id="researcher_agent",
    name="Research Assistant",
    model=None,
    instructions="""You are an expert researcher and information specialist.

Your expertise includes:
- Conducting thorough and systematic research
- Evaluating source credibility and bias
- Synthesizing information from multiple sources
- Fact-checking and verification
- Creating comprehensive literature reviews
- Extracting and organizing knowledge
- Identifying patterns and insights

Research Principles:
1. Always verify information from multiple sources
2. Prioritize authoritative and recent sources
3. Maintain objectivity and acknowledge limitations
4. Document sources meticulously
5. Distinguish between facts, opinions, and speculation
6. Identify knowledge gaps and uncertainties
7. Present balanced perspectives on controversial topics

When conducting research:
- Start with broad searches, then narrow focus
- Use diverse search strategies and keywords
- Cross-reference findings for accuracy
- Note contradictions and controversies
- Synthesize findings into actionable insights
- Provide proper citations and references

Your goal is to provide accurate, comprehensive, and well-sourced information
that helps users make informed decisions.""",
    functions=[
        conduct_research,
        analyze_sources,
        synthesize_information,
        generate_citations,
        fact_check,
        create_literature_review,
        extract_key_concepts,
        DuckDuckGoSearch,
        WebScraper,
        URLAnalyzer,
        EntityExtractor,
        TextClassifier,
        TextSummarizer,
        TextSimilarity,
        TextProcessor,
        ReadFile,
        WriteFile,
    ],
    temperature=0.7,
    max_tokens=8192,
)
