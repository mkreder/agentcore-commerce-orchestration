"""Memory: shared versus isolated.

    SHARED       everyone reads:   service data (specs.py), detector flags, the customer's vehicle
    ADVISOR      advisor only:     the conversation (the advisor agent's own message history)
    COMMERCE     commerce only:    cart and checkout state, store errors, pricing and fitment results
    MONITOR      monitor only:     its verdicts, escalation state. Nothing it can write to the store.

Here it is plain in-process state, keyed by session id. AgentCore Runtime gives every session its own
microVM, so this is enough for a demo. To make it durable, map each block to an AgentCore Memory
namespace: /sessions/{id}/shared, /advisor, /commerce and /monitor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .detector import Measurement, Vehicle


def _settled() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


@dataclass
class SharedState:
    vehicle: Vehicle | None = None  # the car every fitment check and service figure is judged against
    measurement: Measurement | None = None  # the last wear measurement the customer gave
    flags_seen: list[str] = field(default_factory=list)
    turn_flags: tuple[str, ...] = ()  # the turn in progress: regex flags, plus the classifier's when it's done
    flags_settled: asyncio.Event = field(default_factory=_settled)  # set once every detection layer is done


@dataclass
class CartLine:
    variant_id: str
    title: str
    quantity: int
    unit_price: int  # minor units


@dataclass
class CommerceState:
    cart_id: str | None = None
    lines: dict[str, CartLine] = field(default_factory=dict)
    discount_code: str | None = None
    continue_url: str | None = None
    last_cart_action: dict | None = None  # set by cart tools during the current turn
    tool_errors: list[str] = field(default_factory=list)
    turn_requests: list[str] = field(default_factory=list)  # what the advisor asked for this turn
    turn_usage: dict[str, int] = field(default_factory=dict)  # commerce agent tokens this turn

    def start_turn(self) -> None:
        self.last_cart_action, self.turn_requests, self.turn_usage = None, [], {}


@dataclass
class MonitorState:
    escalated: bool = False
    escalation_reason: str | None = None
    verdicts: list[dict] = field(default_factory=list)


@dataclass
class Session:
    id: str
    shared: SharedState = field(default_factory=SharedState)
    commerce: CommerceState = field(default_factory=CommerceState)
    monitor: MonitorState = field(default_factory=MonitorState)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(id=session_id)
        return self._sessions[session_id]
