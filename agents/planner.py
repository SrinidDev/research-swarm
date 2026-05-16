import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import BaseAgent
from orchestrator import Message, Orchestrator, AgentStatus


class PlannerAgent(BaseAgent):

    @property
    def system_prompt(self) -> str:
        return """You are a research planning agent in a multi-agent AI swarm.

Your ONLY job is to decompose a complex research question into concrete subtasks
that can be handled in parallel by specialist agents.

Available agent types:
- retriever: searches the web for information, finds sources
- executor:  runs Python code, does calculations, transforms data
- validator: fact-checks specific claims against sources

Rules:
1. Output ONLY valid JSON — no prose before or after
2. Create 2-4 subtasks (more = slower, fewer = shallower)
3. Each subtask must be specific and actionable
4. Prefer parallel tasks (empty depends_on) when possible
5. Never assign tasks to "planner", "synthesizer", or "critic"

Output schema:
{
  "subtasks": [
    {
      "id": "t1",
      "agent": "retriever" | "executor" | "validator",
      "task": "<specific actionable instruction>",
      "depends_on": []
    }
  ],
  "strategy": "<1-2 sentences explaining the approach>"
}"""

    async def handle(self, msg: Message):
        user_query = msg.payload["query"]

        raw = await self.ask_llm(
            messages=[{"role": "user", "content": user_query}],
            max_tokens=1024,
        )

        try:
            plan = json.loads(self.clean_json(raw))
        except json.JSONDecodeError as e:
            raise ValueError(f"Planner returned invalid JSON: {e}\n\nRaw: {raw}")

        print(f"[planner] strategy: {plan['strategy']}")
        print(f"[planner] {len(plan['subtasks'])} subtasks:")
        for st in plan["subtasks"]:
            print(f"  [{st['id']}] {st['agent']}: {st['task'][:70]}")

        self.orc.store_result("planner", plan)
        await self.orc.set_status(self.name, AgentStatus.DONE, {
            "subtasks": len(plan["subtasks"]),
            "strategy": plan["strategy"],
        })
        await self.orc.send(Message(
            sender=self.name,
            receiver="orchestrator",
            type="plan_ready",
            payload=plan,
            parent_id=msg.id,
        ))


