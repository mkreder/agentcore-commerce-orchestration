"""The nodes of the turn graph, as custom Strands Graph nodes.

    detector ──▶ advisor ────[review gate]──▶ monitor
             └─▶ classifier ─[review gate]──┘

The advisor and the classifier run in parallel. Each node is a MultiAgentBase, so the Graph runs a
regex, an agent with tools and a model call the same way. Nodes don't read the text that Graph passes
along by default (the task plus every upstream output). They read and write a typed `Turn` in
`invocation_state`, so the monitor gets a structured handoff, not the transcript.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from strands.multiagent import MultiAgentBase, MultiAgentResult, Status

from . import specs
from .classifier import Classifier, timed
from .detector import Detection, detect, with_flags
from .gate import GateDecision, review_gate, stream_mode
from .monitor import Handoff, SafetyMonitor, Verdict
from .session import Session

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """What the nodes know about this turn. The orchestrator reads it once the graph is done."""

    session: Session
    message: str
    detection: Detection | None = None  # the regex's
    classifier: dict | None = None  # the classifier's flags and latency, when it ran
    streaming: bool = True
    draft: str = ""
    first_token_at: float | None = None
    gate: GateDecision | None = None
    verdict: Verdict | None = None
    answer: str = ""
    tokens: dict[str, dict] = field(default_factory=dict)

    @property
    def flags(self) -> Detection:
        """Both detection layers: the regex's flags plus any the classifier added."""
        return with_flags(self.detection, (self.classifier or {}).get("flags", []))


def turn_of(invocation_state: dict[str, Any]) -> Turn:
    return invocation_state["turn"]


class TurnNode(MultiAgentBase):
    """A graph node that works on the Turn. Subclasses implement `run` and may yield stream events."""

    def __init__(self, node_id: str):
        super().__init__()
        self.id = node_id

    async def run(self, turn: Turn) -> AsyncIterator[dict]:
        raise NotImplementedError
        yield {}

    async def stream_async(self, task, invocation_state: dict[str, Any] | None = None, **kwargs) -> AsyncIterator[dict]:
        turn = turn_of(invocation_state or {})
        started = time.time()
        async for event in self.run(turn):
            yield event
        usage = turn.tokens.get(self.id, {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0})
        elapsed = round((time.time() - started) * 1000)
        # The output lives in the Turn, so the result carries only usage and timing.
        yield {
            "result": MultiAgentResult(
                status=Status.COMPLETED,
                accumulated_usage=usage,
                execution_count=1,
                execution_time=elapsed,
            )
        }

    async def invoke_async(self, task, invocation_state: dict[str, Any] | None = None, **kwargs) -> MultiAgentResult:
        final = None
        async for event in self.stream_async(task, invocation_state, **kwargs):
            final = event
        return final["result"]


class DetectorNode(TurnNode):
    """01 Detector: regex, zero tokens. Flags, never blocks."""

    def __init__(self, with_classifier: bool = False):
        super().__init__("detector")
        self.with_classifier = with_classifier

    async def run(self, turn: Turn) -> AsyncIterator[dict]:
        shared = turn.session.shared
        turn.detection = detect(turn.message, known_vehicle=shared.vehicle)
        shared.vehicle = turn.detection.vehicle
        shared.measurement = turn.detection.measurement or shared.measurement
        shared.flags_seen.extend(turn.detection.flags)
        shared.turn_flags = tuple(turn.detection.flags)  # the commerce agent's tools read these
        if self.with_classifier and needs_classifier(turn):
            shared.flags_settled.clear()  # the cart lock waits for the classifier
        # Known-risky turns are buffered; everything else streams as the advisor writes it.
        turn.streaming = stream_mode(turn.detection)
        return
        yield {}


class ClassifierNode(TurnNode):
    """01b Second detection layer, in parallel with the advisor. Adds flags, never removes them.

    Bounded: past `timeout_s` it is cut off. A failure or a timeout marks the classifier unavailable,
    and the review gate sends the turn to the monitor. The cart lock keeps the regex's flags.
    """

    def __init__(self, classifier: Classifier, timeout_s: float):
        super().__init__("classifier")
        self.classifier = classifier
        self.timeout_s = timeout_s

    async def run(self, turn: Turn) -> AsyncIterator[dict]:
        shared = turn.session.shared
        vehicle = turn.detection.vehicle.label if turn.detection.vehicle else None
        started = time.perf_counter()
        try:
            flags, usage, latency = await asyncio.wait_for(
                timed(self.classifier, turn.message, vehicle), timeout=self.timeout_s
            )
            added = [f for f in flags if f not in turn.detection.flags]
            turn.classifier = {"flags": flags, "added": added, "latency_ms": latency}
            if usage:
                turn.tokens["classifier"] = usage
            shared.flags_seen.extend(added)
            shared.turn_flags = (*shared.turn_flags, *added)
        except Exception as e:  # noqa: BLE001 - not knowing means review, never "looks fine"
            reason = "timeout" if isinstance(e, asyncio.TimeoutError) else "error"
            if reason == "error":
                log.exception("classifier failed; sending the turn to review")
            latency = round((time.perf_counter() - started) * 1000)
            turn.classifier = {"flags": [], "added": [], "latency_ms": latency, "unavailable": reason}
        finally:
            shared.flags_settled.set()
        return
        yield {}


class AdvisorNode(TurnNode):
    """02 Advisor, which delegates 03 store actions to the commerce agent. Streams its text as it writes.

    It wraps the session's Strands agent instead of adding the agent to the graph directly: the
    advisor keeps its own conversation across turns, and the detector's flags go into its context
    as system-written lines, not as "Inputs from previous nodes".
    """

    def __init__(self, agent):
        super().__init__("advisor")
        self.agent = agent

    async def run(self, turn: Turn) -> AsyncIterator[dict]:
        commerce = turn.session.commerce
        commerce.start_turn()
        new_message = False
        async for event in self.agent.stream_async(_with_context(turn)):
            if "data" in event and event["data"]:
                # Text written before and after a tool call arrives as separate messages.
                delta = ("\n\n" if new_message and turn.draft else "") + event["data"]
                new_message = False
                turn.draft += delta
                turn.first_token_at = turn.first_token_at or time.perf_counter()
                yield {"delta": delta}
            elif "message" in event:
                new_message = True
            elif "result" in event:
                turn.tokens["advisor"] = dict(event["result"].metrics.accumulated_usage)
        if commerce.turn_requests:
            turn.tokens["commerce"] = dict(commerce.turn_usage)
        turn.answer = turn.draft


class MonitorNode(TurnNode):
    """05 Safety monitor: zero tools, a structured handoff in, a verdict out."""

    def __init__(self, monitor: SafetyMonitor):
        super().__init__("monitor")
        self.monitor = monitor

    async def run(self, turn: Turn) -> AsyncIterator[dict]:
        detection, session = turn.flags, turn.session
        year = detection.vehicle.year if detection.vehicle else None
        mentioned = specs.mentioned_products(turn.message, turn.draft)
        measurement = detection.measurement
        handoff = Handoff(
            customer_message=turn.message,
            draft_answer=turn.draft,
            detector_flags=list(detection.flags),
            vehicle=detection.vehicle.label if detection.vehicle else None,
            measurement=(
                {
                    "part": measurement.part,
                    "value": measurement.value,
                    "unit": measurement.unit,
                    "minimum": measurement.minimum,
                    "summary": measurement.summary,
                }
                if measurement
                else None
            ),
            spec_for_year=({h: specs.spec_for_year(h, year) for h in mentioned} or None if year is not None else None),
            specs_allowed=specs.allowed_specs(mentioned, year),
            cart_action=session.commerce.last_cart_action,
        )
        verdict, usage = await self.monitor.review(handoff)
        turn.verdict = verdict
        turn.tokens["monitor"] = dict(usage)
        session.monitor.verdicts.append(verdict.model_dump())
        if verdict.action == "escalate" and not session.monitor.escalated:
            # Sticky for the session: once escalated, the cart stays paused.
            session.monitor.escalated = True
            session.monitor.escalation_reason = ",".join(verdict.reasons) or "escalated"
        if verdict.action != "pass":
            turn.answer = verdict.answer
        return
        yield {}


def needs_classifier(turn: Turn) -> bool:
    """The classifier can only add flags. If the regex already found a critical one, the turn is
    reviewed anyway, so the call would cost a request and change nothing."""
    return not turn.detection.critical


def run_classifier(state, *, invocation_state: dict[str, Any]) -> bool:
    """Condition on the detector -> classifier edge."""
    return needs_classifier(turn_of(invocation_state))


def needs_review(state, *, invocation_state: dict[str, Any]) -> bool:
    """04 Review gate, as the condition on the edges into the monitor. Code decides, not a model."""
    turn = turn_of(invocation_state)
    gate = review_gate(turn.flags, turn.draft)
    if turn.classifier and turn.classifier.get("unavailable"):
        gate = dataclasses.replace(gate, review=True, reasons=(*gate.reasons, f"classifier_{turn.classifier['unavailable']}"))
    turn.gate = gate
    return gate.review


def _with_context(turn: Turn) -> str:
    """Detector output goes into the advisor's context, written by the system, not the customer."""
    detection, session = turn.detection, turn.session
    lines = []
    if detection.flags:
        lines.append(f"detector_flags: {', '.join(detection.flags)}")
    if detection.vehicle:
        lines.append(f"vehicle: {detection.vehicle.label}")
    if detection.measurement:
        lines.append(f"worn_part: {detection.measurement.summary}")
    if session.monitor.escalated:
        lines.append("session_escalated: true (adding to the cart is paused; checkout still works)")
    if not lines:
        return turn.message
    return "<turn_context>\n" + "\n".join(lines) + "\n</turn_context>\n" + turn.message
