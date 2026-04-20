

```
---
name: news-check
description: Spins up multiple agents to gather and aggregate latest global news
version: "1.0"
tags: [news, research, multi-agent, monitoring]
---

# News Check Skill

## Overview
This skill orchestrates multiple specialized agents to gather, verify, and aggregate the latest news from around the world.

## Prerequisites
- Access to web search tools (e.g., `google_search`, `bing_search`)
- Access to news API tools (e.g., `news_api`, `rss_fetcher`)
- Ability to spawn and manage multiple agent instances

## Step-by-Step Instructions

### 1. Initialize Agent Pool
- Spawn 3-5 specialized news agents with the following roles:
  - **Agent 1 (Global News)**: Focus on worldwide major events
  - **Agent 2 (Regional News)**: Focus on regional/continental news
  - **Agent 3 (Tech & Business)**: Focus on technology and business news
  - **Agent 4 (Politics)**: Focus on political developments
  - **Agent 5 (Breaking News)**: Focus on breaking/urgent stories

### 2. Configure Search Parameters
For each agent, configure:
- **Time Range**: Last 24 hours (or last 6 hours for breaking news)
- **Geographic Scope**: Global (adjust based on agent role)
- **Language**: English (primary), with translations noted
- **Source Priority**: Reputable news outlets only (AP, Reuters, BBC, CNN, etc.)

### 3. Execute News Gathering
Each agent should:

1. **Search for News** using appropriate tools:
   ```
   Tool: google_search OR news_api
   Query: "latest news [topic] [date range]"
   Sources: Filter for .com, .org, .gov domains
   ```

2. **Verify Sources**:
   - Check outlet credibility
   - Cross-reference with at least 2 sources for major stories
   - Flag any unverified claims

3. **Extract Key Information**:
   - Headline
   - Publication date/time
   - Source name
   - Summary (1-2 sentences)
   - Link to original article

### 4. Aggregate Results
Collect all findings and consolidate:

1. **Categorize by Topic**:
   - Politics
   - Business/Economy
   - Technology
   - Health/Science
   - Environment
   - Sports
   - Entertainment

2. **Rank by Importance**:
   - Mark breaking news with [BREAKING]
   - Mark major stories with [IMPORTANT]
   - Mark routine updates with [UPDATE]

3. **Remove Duplicates**:
   - Identify same story from multiple sources
   - Keep the most authoritative version

### 5. Output Format
Return results in the following structured format:

```
## 📰 Latest Global News Summary
**Date**: [Current Date/Time]
**Agents Active**: [Number]

### 🔴 Breaking News
- [Headline] | Source: [Name] | Time: [HH:MM] | [Brief Summary]

### 📌 Major Stories
- [Headline] | Source: [Name] | Time: [HH:MM] | [Brief Summary]

### 📊 By Category
**Politics**:
- [Story 1]
- [Story 2]

**Business**:
- [Story 1]
- [Story 2]

**Technology**:
- [Story 1]
- [Story 2]

### 🔗 Source Links
1. [Headline] - [URL]
2. [Headline] - [URL]

### ⚠️ Verification Notes
- [Any unverified claims or conflicting reports]
```

### 6. Cleanup
- Terminate all spawned agents after completion
- Archive search queries for future reference
- Log total number of sources checked

## Error Handling
- If no news found: Report "No significant news detected in the last [time range]"
- If agent fails: Retry with adjusted parameters (max 3 attempts)
- If API rate-limited: Wait 30 seconds and retry

## Performance Targets
- Total execution time: < 5 minutes
- Minimum sources checked: 10
- Minimum unique stories: 5
- Maximum duplicate rate: 20%
```