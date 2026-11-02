"""The orchestrator: a Strands Graph with deterministic edges. One turn, deterministic first.

    detector ──▶ advisor ─────[04 review gate]──▶ monitor
     01 regex │   02 agent                          05 one model call,
              │   └─ 03 commerce agent, a tool         zero tools
              └─▶ classifier ─[04 review gate]──┘
                  01b model, adds flags only, in parallel with the advisor

The graph decides what runs; no model routes the safety path. The edge into the monitor is a Python
condition, so a flagged turn always gets its second look. Inside the advisor node, the advisor decides
when to call the commerce agent (that is conversation); the commerce agent's tools decide in code
whether a store action is allowed, both for safety and for fitment. After the graph, the orchestrator
ships the answer: text already streamed, a `replace` if the monitor repaired it, or the reviewed answer
if the turn was buffered.

Yields events for the widget, sent over SSE by AgentCore Runtime:
    {"type": "text", "delta": "..."}     answer text, streamed as it is generated
    {"type": "replace", "text": "..."}   a streamed answer the monitor repaired after the fact
    {"type": "done", "trace": {...}}     what happened in this turn: path, flags, review, verdict, tokens, latency
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from strands.multiagent import GraphBuilder
from strands.multiagent.graph import Graph

from .advisor import build_advisor
from .classifier import Classifier, make_classifier
from .commerce.store import Store, make_store
from .config import Settings
from .monitor import SafetyMonitor
from .nodes import AdvisorNode, ClassifierNode, DetectorNode, MonitorNode, Turn, needs_review, run_classifier
from .session import Session, SessionRegistry

AdvisorFactory = Callable[[Settings, Store, Session], Any]

TURN_TIMEOUT_S = 180


def build_turn_graph(
    advisor_agent, monitor: SafetyMonitor, classifier: Classifier | None = None, classifier_timeout_s: float = 3.0
) -> Graph:
    builder = GraphBuilder()
    builder.add_node(DetectorNode(with_classifier=classifier is not None), "detector")
    builder.add_node(AdvisorNode(advisor_agent), "advisor")
    builder.add_node(MonitorNode(monitor), "monitor")
    builder.add_edge("detector", "advisor")
    builder.add_edge("advisor", "monitor", condition=needs_review)
    if classifier is not None:
        # Same batch as the advisor, so they run in parallel; the monitor waits for both.
        # Skipped when the regex already found a critical flag: that turn is reviewed anyway.
        builder.add_node(ClassifierNode(classifier, classifier_timeout_s), "classifier")
        builder.add_edge("detector", "classifier", condition=run_classifier)
        builder.add_edge("classifier", "monitor", condition=needs_review)
    builder.set_entry_point("detector")
    builder.set_max_node_executions(4)
    builder.set_execution_timeout(TURN_TIMEOUT_S)
    builder.set_graph_id("parts_turn")
    return builder.build()


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        store: Store | None = None,
        advisor_factory: AdvisorFactory = build_advisor,
        monitor: SafetyMonitor | None = None,
        classifier: Classifier | None | str = "from_settings",
    ):
        self.settings = settings
        self.store = store or make_store(settings)
        self.advisor_factory = advisor_factory
        self.monitor = monitor or SafetyMonitor(settings)
        self.classifier = make_classifier(settings) if classifier == "from_settings" else classifier
        self.sessions = SessionRegistry()
        self._graphs: dict[str, tuple[Any, Graph]] = {}

    def _graph(self, session: Session) -> tuple[Any, Graph]:
        """One advisor and one graph per session: the advisor keeps its conversation across turns."""
        if session.id not in self._graphs:
            advisor = self.advisor_factory(self.settings, self.store, session)
            self._graphs[session.id] = (advisor, build_turn_graph(advisor, self.monitor, self.classifier, self.settings.classifier_timeout_s))
        return self._graphs[session.id]

    async def run_turn(self, session_id: str, message: str) -> AsyncIterator[dict]:
        started = time.perf_counter()
        session = self.sessions.get(session_id)
        advisor, graph = self._graph(session)
        turn = Turn(session=session, message=message)

        async for event in graph.stream_async(message, invocation_state={"turn": turn}):
            if event.get("type") != "multiagent_node_stream" or event["node_id"] != "advisor":
                continue
            delta = event["event"].get("delta")
            # The detector already ran, so the turn knows whether to stream or buffer.
            if delta and turn.streaming:
                yield {"type": "text", "delta": delta}

        answer, draft = turn.answer, turn.draft
        if turn.verdict is not None and answer != draft:
            # The advisor's own history must hold what the customer saw, not the draft.
            _overwrite_last_reply(advisor, answer)

        first_token_at = turn.first_token_at
        if not turn.streaming:
            first_token_at = time.perf_counter()
            yield {"type": "text", "delta": answer}
        elif answer != draft:
            yield {"type": "replace", "text": answer}

        yield {"type": "done", "trace": _trace(session_id, graph, turn, started, first_token_at)}


def _trace(session_id: str, graph: Graph, turn: Turn, started: float, first_token_at: float | None) -> dict:
    review: dict[str, Any] = {"reviewed": turn.gate.review, "reasons": list(turn.gate.reasons)}
    if turn.verdict is not None:
        review["verdict"] = turn.verdict.action
        review["verdict_reasons"] = turn.verdict.reasons
        if turn.answer != turn.draft:
            review["draft"] = turn.draft  # for audits: what the advisor wrote before the repair
    detection = turn.detection
    return {
        "session_id": session_id,
        "path": [node.node_id for node in graph.state.execution_order],
        "flags": list(detection.flags),
        "vehicle": detection.vehicle.label if detection.vehicle else None,
        "measurement": detection.measurement.summary if detection.measurement else None,
        "classifier": turn.classifier,
        "streamed": turn.streaming,
        "commerce_requests": list(turn.session.commerce.turn_requests),
        "cart_action": turn.session.commerce.last_cart_action,
        "review": review,
        "tokens": turn.tokens,
        "escalated": turn.session.monitor.escalated,
        "latency_ms": {
            "first_token": round(((first_token_at or time.perf_counter()) - started) * 1000),
            "total": round((time.perf_counter() - started) * 1000),
        },
    }


def _overwrite_last_reply(advisor, answer: str) -> None:
    messages = getattr(advisor, "messages", None)
    if messages and messages[-1].get("role") == "assistant":
        messages[-1]["content"] = [{"text": answer}]
