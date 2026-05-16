"""
orchestrator.py — The nervous system of the swarm.

Every agent registers here. The orchestrator:
  1. Maintains a message bus (agent_name -> asyncio.Queue)
  2. Dispatches messages between agents
  3. Tracks agent status (idle / working / done / error)
  4. Emits events so the UI can watch in real time
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


# ── Agent states ──────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    IDLE     = "idle"
    WORKING  = "working"
    DONE     = "done"
    ERROR    = "error"
    RETRYING = "retrying"


# ── Message envelope ──────────────────────────────────────────────────────────

@dataclass
class Message:
    """
    Everything that flows through the bus is a Message.
    
    sender:   who sent it (agent name or "user")
    receiver: who should handle it (agent name or "orchestrator")
    type:     what kind of message ("task", "result", "error", "status")
    payload:  the actual content (any dict)
    id:       unique ID so we can trace message chains
    """
    sender:   str
    receiver: str
    type:     str
    payload:  dict
    id:       str = field(default_factory=lambda: f"msg_{time.time_ns()}")
    parent_id: str | None = None   # links replies back to the original task


# ── Event listener type ───────────────────────────────────────────────────────

# Listeners receive (agent_name, new_status, optional_detail)
StatusListener = Callable[[str, AgentStatus, dict], Coroutine]

# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self):
        # agent_name -> asyncio.Queue of Messages
        self._bus: dict[str, asyncio.Queue] = {}

        # agent_name -> current AgentStatus
        self._status: dict[str, AgentStatus] = {}

        # collected results from all agents, keyed by agent name
        self.results: dict[str, Any] = {}

        # async callbacks notified on every status change (used by SSE layer)
        self._listeners: list[StatusListener] = []

        # internal queue for messages addressed to the orchestrator itself
        self._own_queue: asyncio.Queue = asyncio.Queue()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str) -> asyncio.Queue:
        """
        Call this once per agent at startup.
        Returns the agent's personal inbox queue.
        """
        if name in self._bus:
            raise ValueError(f"Agent '{name}' already registered")
        self._bus[name] = asyncio.Queue()
        self._status[name] = AgentStatus.IDLE
        print(f"[orchestrator] registered agent: {name}")
        return self._bus[name]

    def add_listener(self, fn: StatusListener):
        """Register a coroutine to be called on every status change."""
        self._listeners.append(fn)

    # ── Messaging ────────────────────────────────────────────────────────────

    async def send(self, msg: Message):
        """
        Route a message to the right queue.
        Messages to 'orchestrator' land in the orchestrator's own queue.
        """
        target = msg.receiver
        if target == "orchestrator":
            await self._own_queue.put(msg)
        elif target in self._bus:
            await self._bus[target].put(msg)
        else:
            raise KeyError(f"Unknown receiver: '{target}'. Did you register this agent?")

    async def broadcast(self, sender: str, type: str, payload: dict, receivers: list[str]):
        """Send the same message to multiple agents at once."""
        for receiver in receivers:
            await self.send(Message(
                sender=sender,
                receiver=receiver,
                type=type,
                payload=payload,
            ))

    # ── Status tracking ──────────────────────────────────────────────────────

    async def set_status(self, agent: str, status: AgentStatus, detail: dict = {}):
        """
        Update an agent's status and notify all listeners.
        Agents call this themselves to report progress.
        """
        self._status[agent] = status
        print(f"[orchestrator] {agent} → {status.value}  {detail}")
        for listener in self._listeners:
            await listener(agent, status, detail)

    def get_status(self, agent: str) -> AgentStatus:
        return self._status.get(agent, AgentStatus.IDLE)

    def all_done(self, agents: list[str]) -> bool:
        """Returns True when every listed agent has finished."""
        return all(self._status.get(a) in (AgentStatus.DONE, AgentStatus.ERROR)
                   for a in agents)

    # ── Result collection ────────────────────────────────────────────────────

    def store_result(self, agent: str, result: Any):
        """Agents call this when they have output to share."""
        self.results[agent] = result

    # ── Lifecycle helpers ────────────────────────────────────────────────────

    async def wait_for_agents(self, agents: list[str], poll_interval: float = 0.1):
        """
        Async-sleep until all listed agents are DONE or ERROR.
        This is how the orchestrator gates on parallel work finishing.
        """
        while not self.all_done(agents):
            await asyncio.sleep(poll_interval)

    async def run_agent(self, agent_fn, *args, **kwargs):
        """
        Convenience wrapper: runs an agent coroutine and catches top-level errors
        so one failing agent doesn't crash the whole swarm.
        """
        try:
            await agent_fn(*args, **kwargs)
        except Exception as e:
            name = kwargs.get("name", agent_fn.__name__)
            await self.set_status(name, AgentStatus.ERROR, {"error": str(e)})
            raise
