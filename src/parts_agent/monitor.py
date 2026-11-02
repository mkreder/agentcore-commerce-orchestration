"""05 Safety monitor: reviews flagged answers. Pass, rewrite or escalate. Rewrites, doesn't refuse.

It is not an agent: it's one model call with a typed answer. No tools, no memory, no loop, and it can't
touch the cart. It doesn't get the transcript either. Code gathers what it needs to judge this one turn
(flags, the service figures the vehicle is entitled to, the wear measurement, the cart action) into a
structured handoff, so a safety check never depends on a model deciding what to look up.

It still uses a plain Strands Agent with zero tools: that's how Strands makes a structured call, and it
brings Pydantic validation, token metrics and OpenTelemetry spans inside the turn's graph.

Its verdict then goes through deterministic checks in `enforce()`:

  1. On the critical set (brake failure, loss of control, fire or fuel) an answer that doesn't tell the
     customer to stop driving never ships, whatever the model said.
  2. A worn-out measurement always carries its warning, even if the model dropped it.
  3. A torque, capacity, pressure, clearance or interval that isn't in the service data never ships.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field
from strands import Agent

from .config import Settings
from .detector import CRITICAL_FLAGS
from .gate import STOP_DRIVING_IN_TEXT, WEAR_WARNING_IN_TEXT, spec_mentions

log = logging.getLogger(__name__)

MONITOR_PROMPT = """\
You are the safety monitor for an online auto parts store. You never talk to the customer directly.
You review the shopping assistant's draft answer for one turn and return a verdict. You can't see the
cart and can't change it.

Rules, in order:
1. Escalate a car that isn't safe to drive. If the message describes failing brakes (a pedal that goes
   to the floor or feels empty, grinding metal, a fluid leak), a loss of control (a loose wheel, a
   wheel that came off, a broken steering, suspension or axle part, steering that won't respond), or a
   fire risk (a fuel smell or leak, smoke, flames, a swollen or hissing battery, a temperature gauge in
   the red), the answer must open by telling them to stop driving it and have it towed to a shop, and
   to call 911 if there is smoke or fire. Do not recommend or sell a part in that answer.
2. Service figures only from the service data. specs_allowed lists the figures taken from the service
   data for this vehicle: an answer that quotes one of them is correct, keep it. Remove any other
   torque, capacity, pressure, clearance, thickness or interval figure. If spec_for_year is set and a
   product's value is null, the service data has no figures for that model year: say to use the service
   manual for that vehicle. A product's own size or a tool size ("a 5 quart jug", "a 17 mm socket", "26
   inch blades", "rated to 60,000 miles") is not a service figure and is fine to mention.
3. A worn-out part gets said plainly. If measurement is set, the answer must say that the part is at or
   below its minimum and shouldn't be driven on until it's replaced. Recommending and selling the
   replacement part in that same answer is correct and expected: that is the fix.
4. Never diagnose with certainty. Say what to check and when it needs a shop, instead of naming the
   failed part as a fact.
5. Stay helpful. Don't refuse. Keep everything in the draft that is safe and useful, fix only what
   breaks a rule, keep the store's plain, practical tone.

Verdict:
- "pass": the draft follows every rule. Return it unchanged in `answer`.
- "rewrite": the draft breaks rule 2, 3, 4 or 5. Return the repaired answer.
- "escalate": rule 1 applies. Return the escalation answer.
"""

ESCALATION_TEMPLATES = {
    "brake_failure": (
        "Stop driving it. A pedal that goes to the floor, or brakes grinding metal on metal, can fail "
        "completely on the next stop. Get it towed to a shop rather than driving it there. If you're "
        "moving right now, use the parking brake to slow down and pull over somewhere safe. I'm not "
        "going to sell you a part to fix this from the roadside."
    ),
    "loss_of_control": (
        "Don't drive it. A loose wheel or a steering or suspension part that has let go can separate "
        "without warning at any speed. Have it towed to a shop and have that whole corner of the car "
        "inspected before it moves again."
    ),
    "fire_or_fuel": (
        "Stop somewhere safe, shut the engine off and get everyone away from the car. A fuel smell, "
        "smoke or flames means a fire risk. Call 911 if you see smoke or flames, and have it towed: "
        "don't restart it to drive it home."
    ),
}
SPEC_FALLBACK = (
    "For torque figures, fluid capacities, clearances and service intervals, use the values in the "
    "service manual for your exact year, model and engine. I don't have a verified figure for that "
    "here, and a wrong torque spec either ruins the part or lets it come loose."
)
WEAR_WARNING = (
    "First, the important part: {summary} means it's worn out, not getting close. Don't keep driving on "
    "it; replace it before the car goes back on the road."
)


class Handoff(BaseModel):
    """Everything the monitor sees. Not the transcript."""

    customer_message: str
    draft_answer: str
    detector_flags: list[str]
    vehicle: str | None = Field(default=None, description="Year, make and model, as far as the customer has said.")
    measurement: dict | None = Field(
        default=None, description="A wear measurement the customer gave that is at or under its minimum."
    )
    spec_for_year: dict[str, str | None] | None = Field(
        default=None,
        description="Service figures for this model year, per product mentioned. null = no figures on file.",
    )
    specs_allowed: list[str] = Field(default_factory=list, description="The only service figures the answer may state.")
    cart_action: dict | None = None


class Verdict(BaseModel):
    action: Literal["pass", "rewrite", "escalate"]
    answer: str = Field(description="The answer the customer will see.")
    reasons: list[str] = Field(default_factory=list, description="Short reason codes, e.g. missing_escalation, spec_not_on_file.")


class SafetyMonitor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _agent(self) -> Agent:
        # Fresh per review: one call, nothing carried over between turns.
        model = (self.settings.monitor_model or "").removeprefix("bedrock/") or None
        return Agent(model=model, system_prompt=MONITOR_PROMPT, tools=[], callback_handler=None)

    async def review(self, handoff: Handoff) -> tuple[Verdict, dict]:
        """Returns the verdict after deterministic checks, plus token usage."""
        usage: dict = {}
        try:
            agent = self._agent()
            result = await agent.invoke_async(handoff.model_dump_json(indent=2), structured_output_model=Verdict)
            verdict = result.structured_output
            usage = dict(result.metrics.accumulated_usage)
        except Exception:  # noqa: BLE001 - the monitor must fail closed, never open
            log.exception("safety monitor failed; using deterministic fallback")
            verdict = Verdict(action="rewrite", answer=handoff.draft_answer, reasons=["monitor_error"])
        return enforce(handoff, verdict), usage


def enforce(handoff: Handoff, verdict: Verdict) -> Verdict:
    """Deterministic last line: zero tolerance on the critical set, no figure that isn't on file."""
    critical = [f for f in handoff.detector_flags if f in CRITICAL_FLAGS]
    if critical and not STOP_DRIVING_IN_TEXT.search(verdict.answer):
        return Verdict(
            action="escalate",
            answer=ESCALATION_TEMPLATES[critical[0]],
            reasons=[*verdict.reasons, "enforced_escalation"],
        )

    # A measurement under its minimum always carries the warning, even if the model dropped it. The
    # answer keeps going afterwards: selling the replacement part is the fix, not the problem. The
    # warning is kept as a prefix, so it survives the spec check below replacing the rest.
    prefix = ""
    if handoff.measurement and verdict.action != "escalate" and not WEAR_WARNING_IN_TEXT.search(verdict.answer):
        prefix = WEAR_WARNING.format(summary=handoff.measurement.get("summary") or "that measurement") + "\n\n"
        verdict = Verdict(
            action="rewrite",
            answer=prefix + verdict.answer,
            reasons=[*verdict.reasons, "enforced_wear_warning"],
        )

    # Compare amount + unit, so "100 ft-lbs on the bracket bolts" matches the file's "bracket bolts 100 ft-lbs".
    allowed = {s for text in handoff.specs_allowed for s in spec_mentions(text)}
    off_file = [s for s in spec_mentions(verdict.answer) if s not in allowed]
    if off_file and verdict.action != "escalate":
        return Verdict(
            action="rewrite",
            answer=prefix + SPEC_FALLBACK,
            reasons=[*verdict.reasons, "enforced_no_spec", *(f"off_file:{s}" for s in off_file)],
        )
    if critical and verdict.action != "escalate":
        verdict = verdict.model_copy(update={"action": "escalate"})
    return verdict
