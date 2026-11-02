"""Where a chat turn runs, for the UI: the deployed AgentCore Runtime, or the orchestrator in this process.

Both yield the same events as the runtime's SSE stream:
    {"type": "text", "delta": "..."}   {"type": "replace", "text": "..."}
    {"type": "done", "trace": {...}}   {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import Iterator

import boto3


class RuntimeRunner:
    """Calls a deployed AgentCore Runtime and reads its Server-Sent Events."""

    def __init__(self, runtime_arn: str, profile: str | None, region: str | None = None):
        self.arn = runtime_arn
        region = region or runtime_arn.split(":")[3]
        self.client = boto3.Session(profile_name=profile or None, region_name=region).client("bedrock-agentcore")

    def stream(self, session_id: str, message: str) -> Iterator[dict]:
        try:
            response = self.client.invoke_agent_runtime(
                agentRuntimeArn=self.arn,
                runtimeSessionId=session_id,
                payload=json.dumps({"prompt": message}).encode(),
            )
            for line in response["response"].iter_lines():
                if not line.startswith(b"data: "):
                    continue
                event = json.loads(line[len(b"data: "):])
                if "type" not in event:  # an error raised inside the runtime, reported by the AgentCore SDK
                    yield {"type": "error", "message": f"{event.get('error_type', 'error')}: {event.get('error', event)}"}
                else:
                    yield event
        except Exception as e:  # noqa: BLE001 - show it in the chat instead of crashing the page
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


class LocalRunner:
    """Runs the orchestrator in this process, on its own event loop, so a sync UI can stream it."""

    def __init__(self):
        from parts_agent.config import Settings
        from parts_agent.pipeline import Orchestrator

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

        async def make():
            return Orchestrator(Settings.from_env())

        self.orchestrator = asyncio.run_coroutine_threadsafe(make(), self.loop).result()

    def stream(self, session_id: str, message: str) -> Iterator[dict]:
        events: queue.Queue = queue.Queue()

        async def pump():
            try:
                async for event in self.orchestrator.run_turn(session_id, message):
                    events.put(event)
            except Exception as e:  # noqa: BLE001
                events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                events.put(None)

        asyncio.run_coroutine_threadsafe(pump(), self.loop)
        while (event := events.get()) is not None:
            yield event


def turn_graph(trace: dict) -> str:
    """The turn as a Graphviz diagram: what ran is filled, what was skipped is dashed and grey."""
    ran = set(trace.get("path", []))
    commerce = bool(trace.get("commerce_requests"))
    review = trace.get("review", {})
    verdict = review.get("verdict")
    classifier = trace.get("classifier")

    def node(name: str, label: str, active: bool, color: str = "#2563eb") -> str:
        if active:
            return f'{name} [label="{label}", style="filled,rounded", fillcolor="{color}", fontcolor="white", color="{color}"]'
        return f'{name} [label="{label}", style="dashed,rounded", color="#9ca3af", fontcolor="#9ca3af"]'

    clf_label = "Classifier\\n(skipped: regex already critical)" if classifier is None and "classifier" not in ran else "Classifier"
    if classifier and classifier.get("unavailable"):
        clf_label = f"Classifier\\n{classifier['unavailable']} → review"
    monitor_color = {"escalate": "#dc2626", "rewrite": "#d97706"}.get(verdict, "#16a34a")
    monitor_label = f"Safety monitor\\n{verdict}" if verdict else "Safety monitor"
    gate = "review" if review.get("reviewed") else "no review"
    cart = trace.get("cart_action") or {}
    commerce_label = "Commerce agent\\n(tool)"
    if cart.get("fitment"):
        commerce_label = f"Commerce agent\\nfitment: {cart['fitment']}"
    return "\n".join([
        "digraph turn {",
        'rankdir=LR; bgcolor="transparent"; node [shape=box, fontname="Helvetica", fontsize=11]; edge [fontname="Helvetica", fontsize=9, color="#6b7280"];',
        node("msg", "Customer\\nmessage", True, "#374151"),
        node("detector", "Regex detector", "detector" in ran),
        node("classifier", clf_label, "classifier" in ran, "#7c3aed"),
        node("advisor", "Parts advisor", "advisor" in ran),
        node("commerce", commerce_label, commerce, "#0891b2"),
        node("monitor", monitor_label, "monitor" in ran, monitor_color),
        node("out", "Response", True, "#374151"),
        "msg -> detector; detector -> advisor; detector -> classifier [style=dashed];",
        'advisor -> commerce [style=dotted, dir=both, label="as a tool"];',
        f'advisor -> monitor [label="gate: {gate}"]; classifier -> monitor [style=dashed];',
        'advisor -> out [style=dashed, label="stream"]; monitor -> out;',
        "}",
    ])
