---
name: execute-the-news-read-skill-now
description: "Execute the 'news-read' skill now."
version: 0.1.0
tags: [TodoWriteTool, SpawnAgents, delegate_task, SendMessageTool, DuckDuckGoSearch, ExecuteShell, WebScraper, GoogleSearch, RSSReader]
required_tools: [TodoWriteTool, SpawnAgents, delegate_task, SendMessageTool, DuckDuckGoSearch, ExecuteShell, WebScraper, GoogleSearch, RSSReader]
author: Xerxes-skill-authoring
---
# When to use

Apply this skill for tasks similar to: *Execute the 'news-read' skill now.*

# Procedure

1. **TodoWriteTool** — todos=[{"id": "1", "content": "Ga...
2. **SpawnAgents** — agents=[{"name": "news-politics", ..., wait=true
3. **TodoWriteTool** — todos=[{"id": "1", "content": "Ga...
4. **delegate_task** — target_agent=news-technology, task=Return your collected techn...
5. **delegate_task** — target_agent=news-science, task=Return your collected scien...
6. **delegate_task** — target_agent=news-business, task=Return your collected busin...
7. **delegate_task** — target_agent=news-health, task=Return your collected healt...
8. **SendMessageTool** — target=news-technology, message=Return your collected techn...
9. **SendMessageTool** — target=news-science, message=Return your collected scien...
10. **SendMessageTool** — target=news-business, message=Return your collected busin...
11. **SendMessageTool** — target=news-health, message=Return your collected healt...
12. **DuckDuckGoSearch** — query=top technology news today M..., n_results=5, safesearch=moderate, timelimit=d
13. **DuckDuckGoSearch** — query=top science news today May ..., n_results=5, safesearch=moderate, timelimit=d
14. **DuckDuckGoSearch** — query=top business finance news t..., n_results=5, safesearch=moderate, timelimit=d
15. **DuckDuckGoSearch** — query=top health medical news tod..., n_results=5, safesearch=moderate, timelimit=d
16. **ExecuteShell** — command=curl -sSL --compressed -A '..., timeout=30
17. **ExecuteShell** — command=curl -sSL --compressed -A '..., timeout=30
18. **ExecuteShell** — command=curl -sSL --compressed -A '..., timeout=30
19. **ExecuteShell** — command=curl -sSL --compressed -A '..., timeout=30
20. **WebScraper** — url=https://www.bbc.com/news, selector=h3, clean_text=true, timeout=30
21. **WebScraper** — url=https://www.theguardian.com, selector=h3, clean_text=true, timeout=30
22. **GoogleSearch** — query=top technology news today, n_results=5, time_range=d
23. **GoogleSearch** — query=top science news today, n_results=5, time_range=d
24. **GoogleSearch** — query=top business market news today, n_results=5, time_range=d
25. **GoogleSearch** — query=top health medical news today, n_results=5, time_range=d
26. **RSSReader** — feed_url=https://feeds.bbci.co.uk/ne..., max_items=10, include_content=false
27. **RSSReader** — feed_url=https://rss.nytimes.com/new..., max_items=10, include_content=false
28. **TodoWriteTool** — todos=[{"id": "1", "content": "Ga...

# Verification

After running the procedure, the agent should have invoked these tools in order: `TodoWriteTool>SpawnAgents>TodoWriteTool>delegate_task>delegate_task>delegate_task>delegate_task>SendMessageTool>SendMessageTool>SendMessageTool>SendMessageTool>DuckDuckGoSearch>DuckDuckGoSearch>DuckDuckGoSearch>DuckDuckGoSearch>ExecuteShell>ExecuteShell>ExecuteShell>ExecuteShell>WebScraper>WebScraper>GoogleSearch>GoogleSearch>GoogleSearch>GoogleSearch>RSSReader>RSSReader>TodoWriteTool`.
Total successful calls expected: **28**.
Reference final response (truncated): *# 📰 Global News Briefing — April 19, 2026  ---  ## 🏛️ POLITICS  **1. Trump Pushes Sweeping Tax & Spending Cuts Bill Through Congress** President Donald Trump ma*
