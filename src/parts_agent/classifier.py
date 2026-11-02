"""01b Second detection layer: a small model that reads the message the way the regex can't.

The regex (detector.py) is the floor: zero latency, deterministic, and nothing a customer types can
argue with it. It misses paraphrases ("the pedal feels like a sponge"), other languages and typos.
This layer catches those, under one rule: it can only ADD flags, never remove them. A wrong answer
from it costs an extra review, never a missed escalation.

It runs in parallel with the advisor, so it adds no time to first token. Its flags reach the review
gate, the monitor and the commerce agent's cart lock (which waits for it before adding anything).
It is skipped when the regex already found a critical flag: that turn goes to the monitor anyway.
If it fails or runs out of time, the turn goes to the monitor too (nodes.py): not knowing means review.
"""

from __future__ import annotations

import time
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel

from .config import Settings

Flag = Literal["brake_failure", "loss_of_control", "fire_or_fuel", "below_minimum", "spec_question", "diagnosis_request"]

CLASSIFIER_PROMPT = """\
You screen messages that customers send to the chat of an online auto parts store. Messages can be in
any language, use local slang for car parts, and have typos. Return every flag that applies, or none.

- brake_failure: the brakes are failing or gone: a pedal that goes to the floor or feels soft, spongy
  or empty, a pedal that sinks while held, no braking, grinding metal when braking, a brake fluid leak.
- loss_of_control: the car can't be steered or held on the road: a loose wheel, a wheel that came off,
  loose or missing lug nuts, a broken or separated steering, suspension or axle part, a snapped spring,
  steering that has gone loose or won't respond, a tire rubbing the body.
- fire_or_fuel: fire risk: a fuel or petrol smell, fuel leaking or dripping, smoke from under the hood
  or the dash, flames, a burning or electrical smell, a swollen, hissing or venting battery, a
  temperature gauge in the red, steam from the engine, coolant boiling over.
- below_minimum: the customer gives a measurement for a wear part that is worn out: tire tread at or
  under 2/32 in or 1.6 mm, brake friction material at or under 3 mm, a rotor under its discard thickness.
- spec_question: asks for a torque figure, a fluid capacity, a tire pressure, a clearance, a gap, a
  minimum or discard thickness, a tightening sequence, or a service interval.
- diagnosis_request: asks what is wrong with the car, or whether a symptom means a specific part has failed.

Questions about parts, fitment, prices, stock, shipping or the cart get no flags.
When unsure whether a message describes a car that isn't safe to drive, flag it: a false alarm costs a
second look, a miss can put someone on the highway with no brakes.
"""


class Classification(BaseModel):
    flags: list[Flag] = Field(default_factory=list, description="Every flag that applies; empty if none.")


class Classifier(Protocol):
    name: str

    async def classify(self, message: str, vehicle: str | None) -> tuple[list[str], dict]:
        """Returns the flags it found and its token usage. Raises if it can't answer."""
        ...


class ModelClassifier:
    """A Bedrock model with a typed answer: one call, no tools (like the monitor).

    The model client is built once and reused, so each call skips the connection setup. Each call
    still gets a fresh Agent: nothing carries over between messages.
    """

    name = "model"

    def __init__(self, model_id: str | None, reasoning_effort: str | None = None):
        self.model_id = (model_id or "").removeprefix("bedrock/") or None
        self.model = BedrockModel(model_id=self.model_id, **_request_fields(self.model_id, reasoning_effort))

    async def classify(self, message: str, vehicle: str | None) -> tuple[list[str], dict]:
        prompt = f"vehicle: {vehicle or 'unknown'}\n<message>\n{message}\n</message>"
        agent = Agent(model=self.model, system_prompt=CLASSIFIER_PROMPT, tools=[], callback_handler=None)
        result = await agent.invoke_async(prompt, structured_output_model=Classification)
        return list(dict.fromkeys(result.structured_output.flags)), dict(result.metrics.accumulated_usage)


def make_classifier(settings: Settings) -> Classifier | None:
    if settings.classifier == "off":
        return None
    return ModelClassifier(settings.classifier_model, settings.classifier_reasoning_effort)


def _request_fields(model_id: str | None, reasoning_effort: str | None) -> dict:
    """OpenAI models on Bedrock take a reasoning effort; "none" skips reasoning, which halves latency
    for a task this small. Other models would reject the field, so it's only sent to OpenAI models."""
    if reasoning_effort and model_id and "openai." in model_id:
        return {"additional_request_fields": {"reasoning": {"effort": reasoning_effort}}}
    return {}


async def timed(classifier: Classifier, message: str, vehicle: str | None) -> tuple[list[str], dict, int]:
    started = time.perf_counter()
    flags, usage = await classifier.classify(message, vehicle)
    return flags, usage, round((time.perf_counter() - started) * 1000)
