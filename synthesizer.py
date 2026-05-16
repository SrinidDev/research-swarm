import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from orchestrator import Message, Orchestrator, AgentStatus


class SynthesizerAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a synthesis agent in a multi-agent AI research swarm.
You receive findings from multiple retriever agents and produce a single, clear answer.
Cite sources using [Source: URL] inline. Output ONLY valid JSON."""


    async def handle(self, msg: Message):
        query        = msg.payload["query"]
        findings_list = msg.payload["findings"]

        findings_text = ""
        for f in findings_list:
            findings_text += f"\n=== Subtask: {f.get('task','?')} ===\n"
            findings_text += f"Findings: {f.get('findings','')}\n"
            findings_text += f"Key facts: {json.dumps(f.get('key_facts',[]))}\n"
            findings_text += f"Sources: {', '.join(f.get('sources',[]))}\n"

        raw = await self.ask_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    f"Research findings:\n{findings_text}\n\n"
                    "Synthesise a complete answer. Output JSON:\n"
                    "{\n"
                    '  "answer": "<full markdown answer with inline citations>",\n'
                    '  "word_count": <integer>,\n'
                    '  "sources_used": ["url1", "url2"],\n'
                    '  "confidence": 0.0-1.0,\n'
                    '  "caveats": "<limitations or empty string>"\n'
                    "}\n"
                    "Output ONLY the JSON."
                )
            }],
            max_tokens=2048,
        )

        try:
            result = json.loads(self.clean_json(raw))
        except json.JSONDecodeError:
            result = {
                "answer":       raw,
                "word_count":   len(raw.split()),
                "sources_used": [],
                "confidence":   0.6,
                "caveats":      "JSON parse failed — raw answer returned",
            }

        print(f"[synthesizer] {result.get('word_count','?')} words, confidence={result.get('confidence','?')}")

        self.orc.store_result("synthesizer", result)
        await self.orc.set_status(self.name, AgentStatus.DONE, {
            "word_count": result.get("word_count"),
            "confidence": result.get("confidence"),
        })
        await self.reply(msg, result)


