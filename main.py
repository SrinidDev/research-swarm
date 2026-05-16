import asyncio
import json
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from swarm import ResearchSwarm

app = FastAPI(title="Research Swarm API")

# Allow the HTML file to call this API from any origin
# (needed when opening index.html directly in the browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request model ─────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}
    
@app.get("/", response_class=HTMLResponse)
async def root():
           return HTMLResponse(conent=open("ui/index.html").read())

# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/research")
async def research(request: ResearchRequest):
    """
    Runs the full swarm pipeline and streams events back to the browser.

    Every event is a JSON string in SSE format:
        data: {"type": "status", "agent": "planner", "status": "working"}\n\n
        data: {"type": "status", "agent": "planner", "status": "done"}\n\n
        data: {"type": "answer", "answer": "...", "score": 0.82}\n\n
        data: {"type": "error",  "message": "..."}\n\n
    """

    async def event_stream():
        # Queue that collects events from the swarm and sends them to browser
        # We use a queue because the swarm runs in async tasks and we need
        # to yield events one at a time to the SSE stream
        event_queue: asyncio.Queue = asyncio.Queue()

        # This callback is called by the orchestrator every time
        # any agent changes status — we just drop it in the queue
        async def on_status_change(agent: str, status, detail: dict):
            await event_queue.put({
                "type":   "status",
                "agent":  agent,
                "status": status.value,
                "detail": detail,
            })

        async def run_swarm():
            try:
                swarm  = ResearchSwarm(status_callback=on_status_change)
                result = await swarm.run(request.query)
                # Push the final answer into the queue when done
                await event_queue.put({
                    "type":     "answer",
                    "answer":   result["answer"],
                    "sources":  result["sources"],
                    "score":    result["score"],
                    "attempts": result["attempts"],
                    "caveats":  result.get("caveats", ""),
                    "plan":     result["plan"],
                })
            except Exception as e:
                await event_queue.put({
                    "type":    "error",
                    "message": str(e),
                })
            finally:
                # Sentinel value — tells the stream loop to stop
                await event_queue.put(None)

        # Start the swarm as a background task
        # It runs concurrently while we yield events from the queue
        swarm_task = asyncio.create_task(run_swarm())

        # Keep yielding events until the sentinel (None) arrives
        while True:
            event = await event_queue.get()

            if event is None:
                # Swarm finished — send a done signal and stop
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break

            # Format as SSE: "data: <json>\n\n"
            yield f"data: {json.dumps(event)}\n\n"

        await swarm_task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Tell browser not to cache SSE responses
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # important for nginx proxies
        },
    )
