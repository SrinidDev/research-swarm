import asyncio
import json
import sys
import os

from orchestrator import Orchestrator, Message, AgentStatus
from agents.planner     import PlannerAgent
from agents.retriever   import RetrieverAgent
from agents.validator   import ValidatorAgent
from agents.synthesizer import SynthesizerAgent
from agents.critic      import CriticAgent
from dotenv import load_dotenv
load_dotenv()
MAX_RETRIES = 2

class ResearchSwarm:
    def __init__(self, status_callback=None):
        self.orc = Orchestrator()

        # One instance of each agent type
        # (In production you'd pool multiple retrievers for true parallelism)
        self.planner     = PlannerAgent("planner", self.orc)
        self.retriever   = RetrieverAgent("retriever", self.orc)
        self.validator   = ValidatorAgent("validator", self.orc)
        self.synthesizer = SynthesizerAgent("synthesizer", self.orc)
        self.critic      = CriticAgent("critic", self.orc)

        if status_callback:
            self.orc.add_listener(status_callback)

    async def run(self, query: str) -> dict:
        """
        Full swarm pipeline. Returns the final answer dict.
        """
        # ── Start all agents as background tasks ──────────────────────────────
        tasks = [
            asyncio.create_task(self.planner.run(),     name="planner"),
            asyncio.create_task(self.retriever.run(),   name="retriever"),
            asyncio.create_task(self.validator.run(),   name="validator"),
            asyncio.create_task(self.synthesizer.run(), name="synthesizer"),
            asyncio.create_task(self.critic.run(),      name="critic"),
        ]

        # ── Step 1: Planning ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"SWARM STARTING: {query}")
        print(f"{'='*60}\n")

        plan_msg = Message(
            sender="user",
            receiver="planner",
            type="task",
            payload={"query": query},
        )
        await self.orc.send(plan_msg)

        # Wait for the plan (blocks until planner sends "plan_ready" to orchestrator)
        plan_result = await self.orc._own_queue.get()
        plan = plan_result.payload
        subtasks = plan["subtasks"]

        # ── Step 2: Dispatch parallel subtasks ────────────────────────────────
        # For simplicity in this version we run all retriever subtasks through
        # the single retriever agent sequentially.
        # In production: spawn one RetrieverAgent per subtask for true parallelism.

        retriever_tasks = [st for st in subtasks if st["agent"] == "retriever"]
        all_findings = []

        # Run retriever subtasks — gather them concurrently via asyncio
        async def run_subtask(subtask):
            msg = Message(
                sender="orchestrator",
                receiver="retriever",
                type="task",
                payload={
                    "task": subtask["task"],
                    "task_id": subtask["id"],
                },
            )
            await self.orc.send(msg)
            # Wait for the result on the orchestrator's queue
            result_msg = await self.orc._own_queue.get()
            return result_msg.payload

        # Temporarily reroute retriever replies to orchestrator queue
        # by patching the retriever's reply target
        original_reply = self.retriever.reply
        async def reply_to_orc(original, payload):
            await self.orc.send(Message(
                sender="retriever",
                receiver="orchestrator",
                type="result",
                payload=payload,
                parent_id=original.id,
            ))
        self.retriever.reply = reply_to_orc

        # Similarly for validator, synthesizer, critic
        for agent in [self.validator, self.synthesizer, self.critic]:
            async def make_reply(a=agent):
                async def reply_to_orc_generic(original, payload):
                    await self.orc.send(Message(
                        sender=a.name,
                        receiver="orchestrator",
                        type="result",
                        payload=payload,
                        parent_id=original.id,
                    ))
                return reply_to_orc_generic
            agent.reply = await make_reply()

        print(f"\n[swarm] dispatching {len(retriever_tasks)} retriever subtask(s)...\n")
        for st in retriever_tasks:
            finding = await run_subtask(st)
            all_findings.append(finding)
            print(f"[swarm] received finding for {st['id']}")

        # ── Step 3: Validate ──────────────────────────────────────────────────
        print("\n[swarm] validating findings...\n")
        await self.orc.send(Message(
            sender="orchestrator",
            receiver="validator",
            type="task",
            payload={"findings": all_findings},
        ))
        val_msg = await self.orc._own_queue.get()
        validation = val_msg.payload

        # ── Step 4: Synthesize + Critic loop ──────────────────────────────────
        final_answer = None
        for attempt in range(1, MAX_RETRIES + 2):
            print(f"\n[swarm] synthesis attempt {attempt}...\n")
            await self.orc.send(Message(
                sender="orchestrator",
                receiver="synthesizer",
                type="task",
                payload={"query": query, "findings": all_findings},
            ))
            syn_msg = await self.orc._own_queue.get()
            synthesis = syn_msg.payload

            print(f"\n[swarm] critic reviewing attempt {attempt}...\n")
            await self.orc.send(Message(
                sender="orchestrator",
                receiver="critic",
                type="task",
                payload={
                    "query":      query,
                    "synthesis":  synthesis,
                    "validation": validation,
                    "attempt":    attempt,
                },
            ))
            crit_msg = await self.orc._own_queue.get()
            decision = crit_msg.payload

            if decision["decision"] == "approve" or attempt > MAX_RETRIES:
                final_answer = {
                    "query":    query,
                    "answer":   synthesis.get("answer", ""),
                    "sources":  synthesis.get("sources_used", []),
                    "score":    decision.get("score", 0),
                    "attempts": attempt,
                    "caveats":  synthesis.get("caveats", ""),
                    "plan":     plan,
                }
                break

            print(f"[swarm] critic says retry: {decision.get('retry_instructions','')[:80]}")
            await self.orc.set_status("critic", AgentStatus.RETRYING, {
                "attempt": attempt,
                "reason": decision.get("retry_instructions", ""),
            })

        # ── Shutdown ──────────────────────────────────────────────────────────
        for name in ["planner", "retriever", "validator", "synthesizer", "critic"]:
            await self.orc.send(Message(
                sender="orchestrator", receiver=name,
                type="shutdown", payload={},
            ))

        for t in tasks:
            try:
                await asyncio.wait_for(t, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                t.cancel()

        return final_answer


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main():
    if len(sys.argv) < 2:
        query = "What are the top 3 open-source LLM frameworks for building AI agents in 2025?"
    else:
        query = " ".join(sys.argv[1:])

    swarm = ResearchSwarm()
    result = await swarm.run(query)

    print(f"\n{'='*60}")
    print("FINAL ANSWER")
    print(f"{'='*60}\n")
    print(result["answer"])
    print(f"\nSources: {', '.join(result['sources'])}")
    print(f"Score: {result['score']}  |  Attempts: {result['attempts']}")
    if result.get("caveats"):
        print(f"Caveats: {result['caveats']}")


if __name__ == "__main__":
    asyncio.run(main())
