"""
tools/web_search.py — Web search with DuckDuckGo (+ mock fallback for testing).

On your local machine, DuckDuckGo will work with no API key.
Install:  pip install ddgs

If you want more reliable results for a hackathon demo, also consider:
  - SerpApi (100 free searches/month): https://serpapi.com
  - Brave Search API (2000 free/month): https://api.search.brave.com

Set SEARCH_MOCK=1 to use offline mock data (useful for unit tests).
"""

import os


def search(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a web search. Returns list of {title, url, snippet}.
    Uses DuckDuckGo by default. Falls back to mock data if env says so.
    """
    if os.environ.get("SEARCH_MOCK") == "1":
        return _mock_search(query, max_results)

    return _ddg_search(query, max_results)


def search_text(query: str, max_results: int = 5) -> str:
    """Same as search() but returns a formatted string for LLM prompts."""
    results = search(query, max_results)
    if not results:
        return f"No results found for: {query}"
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}")
        lines.append(f"    {r['url']}")
        lines.append(f"    {r['snippet']}")
        lines.append("")
    return "\n".join(lines)


def _ddg_search(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo search — works on any machine with internet access."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        print(f"[web_search] DDG failed for '{query}': {e}")
        return []


def _mock_search(query: str, max_results: int) -> list[dict]:
    """
    Offline mock results — for testing without internet.
    Returns plausible-looking results based on keywords in the query.
    """
    q = query.lower()

    base = [
        {
            "title":   f"Overview of {query}",
            "url":     f"https://en.wikipedia.org/wiki/{query.replace(' ','_')}",
            "snippet": f"This article provides a comprehensive overview of {query}, "
                       f"covering key concepts, recent developments, and practical applications."
        },
        {
            "title":   f"{query} — latest developments 2025",
            "url":     f"https://arxiv.org/abs/2501.{hash(query)%99999:05d}",
            "snippet": f"Recent research on {query} demonstrates significant improvements "
                       f"in performance, scalability, and ease of deployment."
        },
        {
            "title":   f"Top tools for {query}",
            "url":     f"https://github.com/topics/{query.replace(' ','-')}",
            "snippet": f"A curated list of open-source tools related to {query}, "
                       f"including benchmarks, tutorials, and community resources."
        },
    ]

    if "llm" in q or "agent" in q or "ai" in q:
        base.insert(0, {
            "title":   "LangGraph, CrewAI, AutoGen — agent framework comparison 2025",
            "url":     "https://blog.langchain.dev/agent-frameworks-2025",
            "snippet": "LangGraph offers fine-grained control for stateful agent workflows. "
                       "CrewAI simplifies multi-agent coordination with role-based agents. "
                       "AutoGen (Microsoft) excels at code generation and conversational agents."
        })

    if "groq" in q:
        base.insert(0, {
            "title":   "Groq inference — benchmarks and models",
            "url":     "https://console.groq.com/docs/models",
            "snippet": "Groq's LPU delivers up to 750 tokens/sec on Llama 3.3 70B. "
                       "Available models include llama-3.3-70b-versatile, mixtral-8x7b-32768, "
                       "and gemma2-9b-it."
        })

    return base[:max_results]
