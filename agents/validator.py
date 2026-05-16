import json
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from orchestrator import Message, Orchestrator, AgentStatus
from tools.web_search import search_text


class ValidatorAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a fact-checking agent in a multi-agent AI research swarm.
You receive research findings and verify key claims with additional web searches.
Output ONLY valid JSON."""

    async def handle(self, msg: Message):
        findings_list = msg.payload["findings"]

        # Collect top facts to check
        facts = [
            fact
            for f in findings_list
            for fact in f.get("key_facts", [])[:2]
        ][:6]  # cap at 6 total facts

        if not facts:
            report = {
                "verdicts": [],
                "trust_score": 0.7,
                "recommendation": "approve",
                "notes": "No key facts to validate.",
            }
        else:
            # Search for verification of key facts
            search_tasks = [asyncio.to_thread(search_text, fact, 3) for fact in facts[:3]]
            search_results = await asyncio.gather(*search_tasks)

            combined = ""
            for fact, result in zip(facts[:3], search_results):
                combined += f"\n\nChecking: {fact}\n{result}"

            raw = await self.ask_llm(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Claims to fact-check:\n" +
                        "\n".join(f"- {f}" for f in facts) +
                        f"\n\nAdditional web evidence:\n{combined}\n\n"
                        "Output a validation report as JSON:\n"
                        "{\n"
                        '  "verdicts": [{"claim": "...", "status": "confirmed|unverified|contradicted", "evidence": "..."}],\n'
                        '  "trust_score": 0.0-1.0,\n'
                        '  "recommendation": "approve|retry|reject",\n'
                        '  "notes": "<overall assessment>"\n'
                        "}\n"
                        "Output ONLY the JSON."
                    )
                }],
                max_tokens=1024,
            )

            try:
                report = json.loads(self.clean_json(raw))
            except json.JSONDecodeError:
                report = {
                    "verdicts": [],
                    "trust_score": 0.6,
                    "recommendation": "approve",
                    "notes": raw,
                }

        print(f"[validator] trust={report.get('trust_score')} rec={report.get('recommendation')}")

        self.orc.store_result("validator", report)
        await self.orc.set_status(self.name, AgentStatus.DONE, {
            "trust_score":    report.get("trust_score"),
            "recommendation": report.get("recommendation"),
        })
        await self.reply(msg, report)


