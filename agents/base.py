import os
import asyncio
from abc import ABC, abstractmethod

from groq import AsyncGroq

from orchestrator import Message, Orchestrator, AgentStatus

# Change this to any model available on Groq:
# https://console.groq.com/docs/models
MODEL = "llama-3.3-70b-versatile"

class BaseAgent(ABC):
    """
    All swarm agents inherit from this.

    Subclasses must implement:
      - system_prompt (property) → str
      - handle(msg: Message)     → the agent's core logic per message
    """

    def __init__(self, name: str, orchestrator: Orchestrator):
        self.name = name
        self.orc = orchestrator
        self.inbox: asyncio.Queue = orchestrator.register(name)
        # One shared async Groq client per agent instance
        self._groq = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", ""))
    
    # ── Run loop ──────────────────────────────────────────────────────────────

    async def run(self):
        """
        The agent's main loop.
        Sits idle until a message arrives, then calls handle().
        Exits cleanly on a 'shutdown' message.
        """
        print(f"[{self.name}] ready")
        while True:
            msg = await self.inbox.get()
            if msg.type == "shutdown":
                break
            await self.orc.set_status(self.name, AgentStatus.WORKING,
                                      {"task": msg.payload.get("task", "")[:60]})
            try:
                await self.handle(msg)
            except Exception as e:
                await self.orc.set_status(self.name, AgentStatus.ERROR, {"error": str(e)})
                print(f"[{self.name}] ERROR: {e}")

    # ── Groq API call ─────────────────────────────────────────────────────────

    async def ask_llm(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.3,   # low = more deterministic JSON output
    ) -> str:
        """
        Call Groq and return the response text directly.

        Why return a string instead of the raw response?
          Groq's SDK already parses the response. Unlike Anthropic's API
          which returns content blocks, Groq follows OpenAI format:
          response.choices[0].message.content is always a plain string.
          Returning the string keeps agent code simple.
        """
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = await self._groq.chat.completions.create(
            model=MODEL,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    
    # ── Helpers ───────────────────────────────────────────────────────────────

    def clean_json(self, raw: str) -> str:
        """
        Strip markdown code fences that LLMs sometimes wrap JSON in.
        e.g.  ```json\n{...}\n```  →  {...}
        """
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            # parts[1] is the content between first ``` pair
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    async def reply(self, original: Message, payload: dict):
        """Convenience: send a result back to whoever sent us a task."""
        await self.orc.send(Message(
            sender=self.name,
            receiver=original.sender,
            type="result",
            payload=payload,
            parent_id=original.id,
        ))
