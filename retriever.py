"""
agents/retriever.py — Web search worker (Groq + DuckDuckGo).

How it works (without built-in tool use):
  1. Ask the LLM: "given this task, what 2-3 search queries should I run?"
  2. Run those queries via DuckDuckGo (free, no key)
  3. Feed ALL search results back to the LLM: "summarise these into findings"
  4. Parse the structured JSON output

This is "plan-then-execute" tool use — the LLM plans the searches,
we execute them, then the LLM reasons over the results.
"""

import json
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from orchestrator import Message, Orchestrator, AgentStatus
from tools.web_search import search_text


class RetrieverAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a research retrieval agent in a multi-agent AI swarm.
You are given research tasks and must search the web to find accurate information.
Always output valid JSON only — no prose before or after."""

    async def handle(self, msg: Message):
        task    = msg.payload["task"]
        task_id = msg.payload.get("task_id", "?")

        print(f"[retriever:{task_id}] starting: {task[:70]}...")

        # Step A: Ask LLM what to search for
        query_response = await self.ask_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"Research task: {task}\n\n"
                    "Output 2-3 specific web search queries to find this information.\n"
                    'Output ONLY a JSON array of strings, e.g.: ["query one", "query two"]'
                )
            }],
            max_tokens=256,
            temperature=0.2,
        )

        try:
            queries = json.loads(self.clean_json(query_response))
            if not isinstance(queries, list):
                queries = [task]
        except Exception:
            queries = [task]

        print(f"[retriever:{task_id}] running {len(queries)} searches: {queries}")

        # Step B: Run searches concurrently
        search_tasks = [asyncio.to_thread(search_text, q, 4) for q in queries]
        raw_results  = await asyncio.gather(*search_tasks)

        combined = ""
        for q, result in zip(queries, raw_results):
            combined += f"\n\n--- Search: {q} ---\n{result}"

        # Step C: Ask LLM to synthesise
        synthesis_response = await self.ask_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"Original research task: {task}\n\n"
                    f"Web search results:\n{combined}\n\n"
                    "Based ONLY on these search results, output your findings as JSON:\n"
                    "{\n"
                    '  "findings": "<2-4 sentences summarising what you found>",\n'
                    '  "key_facts": ["fact 1", "fact 2", "fact 3"],\n'
                    '  "sources": ["url1", "url2"],\n'
                    '  "confidence": 0.0-1.0,\n'
                    '  "gaps": "<what you could not find, or empty string>"\n'
                    "}\n"
                    "Output ONLY the JSON."
                )
            }],
            max_tokens=1024,
            temperature=0.2,
        )

        try:
            findings = json.loads(self.clean_json(synthesis_response))
        except json.JSONDecodeError:
            findings = {
                "findings": synthesis_response,
                "key_facts": [],
                "sources": queries,
                "confidence": 0.4,
                "gaps": "JSON parse failed",
            }

        findings["task_id"] = task_id
        findings["task"]    = task

        print(f"[retriever:{task_id}] done — confidence={findings.get('confidence','?')}")

        self.orc.store_result(f"retriever_{task_id}", findings)
        await self.orc.set_status(self.name, AgentStatus.DONE, {
            "task_id":    task_id,
            "confidence": findings.get("confidence", 0),
        })
        await self.reply(msg, findings)
