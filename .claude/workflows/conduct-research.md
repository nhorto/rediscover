---
description: Multi-source parallel research orchestration using available researcher agents
---

# Conduct Research Workflow

Orchestrate comprehensive multi-source research by launching parallel researcher agents.

## Three Research Modes

| Mode | Agents per Type | Timeout | When |
|------|----------------|---------|------|
| Quick | 1 | 2 min | Simple queries |
| Standard | 3 | 3 min | Default |
| Extensive | 8 | 10 min | Deep research |

## Workflow

### 1. Decompose Question
Break the research question into focused sub-questions (N per researcher type, where N = mode agent count).

### 2. Launch All Agents in Parallel
Use a SINGLE message with multiple Agent tool calls:

```
Agent(subagent_type="researcher", prompt="Sub-question 1...")
Agent(subagent_type="claude-researcher", prompt="Sub-question 2...")
Agent(subagent_type="perplexity-researcher", prompt="Sub-question 3...")
Agent(subagent_type="gemini-researcher", prompt="Sub-question 4...")
```

### 3. Collect Results (respect timeouts)
After timeout, proceed with whatever results have returned.

### 4. Synthesize
- Identify common themes (high confidence)
- Unique insights from each source (medium confidence)
- Conflicting information (flag)
- Calculate coverage metrics

## Critical Rules
- Launch ALL agents in ONE message (parallel execution)
- Each agent gets ONE focused sub-question
- TIMELY RESULTS > PERFECT COMPLETENESS
- After timeout, synthesize with what you have

## Handling Blocked Content
If agents report being blocked:
- Use four-tier-scrape workflow for specific URLs
- `mcp__Brightdata__scrape_as_markdown` for bot-protected sites
- `mcp__Brightdata__search_engine` for search with CAPTCHA bypass
