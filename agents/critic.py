import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from orchestrator import Message, Orchestrator, AgentStatus


class CriticAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a critic agent in a multi-agent AI research swarm.
You evaluate synthesised answers for quality, accuracy, and completeness.
Be tough but fair. Output ONLY valid JSON."""

    async def handle(self, msg: Message):
        query          = msg.payload["query"]
        synthesis      = msg.payload["synthesis"]
        validation     = msg.payload["validation"]
        attempt_number = msg.payload.get("attempt", 1)

        raw = await self.ask_llm(
            messages=[{
                "role": "user",
                "content": (
                    f"Original query: {query}\n\n"
                    f"Synthesized answer:\n{synthesis.get('answer','')}\n\n"
                    f"Validator report: trust_score={validation.get('trust_score','?')}, "
                    f"recommendation={validation.get('recommendation','?')}, "
                    f"notes={validation.get('notes','')}\n\n"
                    f"This is attempt #{attempt_number}.\n\n"
                    "Evaluate and output JSON:\n"
                    "{\n"
                    '  "decision": "approve|retry|reject",\n'
                    '  "score": 0.0-1.0,\n'
                    '  "strengths": ["..."],\n'
                    '  "weaknesses": ["..."],\n'
                    '  "retry_instructions": "<if retry: what to redo; else empty>",\n'
                    '  "final_notes": "<brief assessment>"\n'
                    "}\n"
                    "Decision guide: approve if score >= 0.65, retry if 0.4-0.64, reject if < 0.4.\n"
                    "Output ONLY the JSON."
                )
            }],
            max_tokens=512,
        )

        try:
            decision = json.loads(self.clean_json(raw))
        except json.JSONDecodeError:
            decision = {
                "decision":          "approve",
                "score":             0.6,
                "strengths":         [],
                "weaknesses":        ["critic parse error"],
                "retry_instructions": "",
                "final_notes":       raw,
            }

        print(f"[critic] decision={decision['decision']} score={decision.get('score')}")

        self.orc.store_result("critic", decision)
        await self.orc.set_status(self.name, AgentStatus.DONE, {
            "decision": decision["decision"],
            "score":    decision.get("score"),
        })
        await self.reply(msg, decision)

