"""01 Detector: regex over the customer's message. Flags, never blocks.

A car that isn't safe to drive has to be caught deterministically and auditably, so this is not a
model. It costs zero tokens and runs on every turn. Read the patterns below: that is the whole
trigger list.

It also pulls two things out of the message that the rest of the turn needs:

- the **vehicle** (year, make, model), carried across turns, which gates both fitment and every
  service figure the answer is allowed to quote;
- any **measurement** the customer gives for a wear part, compared in code against the minimums
  below. "my tread is 1/32" is not a judgement call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# Flags that always send the turn to the safety monitor and must end in an escalation: the car is
# not safe to drive, and no part we sell changes that today.
CRITICAL_FLAGS = frozenset({"brake_failure", "loss_of_control", "fire_or_fuel"})
# Flags that send the turn to the monitor, which checks service figures and diagnosis language.
REVIEW_FLAGS = CRITICAL_FLAGS | {"below_minimum", "spec_question", "diagnosis_request"}

# Legal or replace-now minimums. At or under these, the part is worn out, whatever else is true.
TREAD_MIN_32NDS = 2.0  # 2/32 in, the legal minimum in most US states
TREAD_MIN_MM = 1.6  # the same figure in mm
PAD_MIN_MM = 3.0  # friction material left on a brake pad

_I = re.IGNORECASE

# Makes, longest first, so "mercedes-benz" wins over "mercedes".
_MAKES = (
    "mercedes-benz|mercedes|land rover|range rover|alfa romeo|volkswagen|chevrolet|mitsubishi|lamborghini"
    "|lincoln|cadillac|chrysler|infiniti|maserati|porsche|subaru|hyundai|genesis|bentley|peugeot|renault"
    "|citroen|acura|nissan|toyota|honda|mazda|lexus|jaguar|buick|dodge|jeep|kia|ram|audi|bmw|ford|mini"
    "|fiat|seat|skoda|volvo|tesla|rivian|gmc|chevy|vw|opel"
)
_VEHICLE = re.compile(
    r"\b(?:(?P<y1>19[5-9]\d|20[0-4]\d)\s+)?"
    rf"(?P<make>{_MAKES})\b"
    r"(?:\s+(?P<model>[A-Za-z][\w-]{1,14}))?"
    r"(?:[\s,]+(?P<y2>19[5-9]\d|20[0-4]\d))?",
    _I,
)
# A bare year, for follow-ups like "mine's a 2012" once the make is already known.
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
# Words that follow a make without being a model.
_NOT_A_MODEL = frozenset(
    "and or for the a an is was it my your that this these those parts part oem dealer please so but "
    "with without stuff thing things one two both all any some do does did can could would should "
    "fits fit dealership".split()
)

_BRAKE_FAILURE = re.compile(
    r"\b("
    r"(brake\s+)?pedal\b.{0,30}\b(to\s+the\s+floor|on\s+the\s+floor|sinks?|sinking|sank|went\s+down|goes?\s+down"
    r"|gives?\s+out|gave\s+out|no\s+resistance|hits?\s+the\s+floor)"
    r"|(to\s+the\s+floor|on\s+the\s+floor)\b.{0,20}\bpedal"
    r"|no\s+brakes\b|brakes?\s+(don'?t|do\s+not|won'?t|will\s+not|didn'?t|aren'?t)\s+(work|working|stop|stopping|hold|holding)"
    r"|lost\s+(the\s+|my\s+|all\s+)?brakes?\b|brake\s+failure|lost\s+braking"
    r"|metal\s+on\s+metal|grinding\b.{0,25}\bbrak|brak\w*\b.{0,25}\bgrinding"
    r"|brake\s+(fluid\s+)?(line\s+)?(leak|leaking|blew|burst|ruptured)|lost\s+(all\s+)?(my\s+|the\s+)?brake\s+fluid"
    r")",
    _I,
)
_LOSS_OF_CONTROL = re.compile(
    r"\b("
    r"wheel\s+(came|fell)\s+off|lost\s+a\s+wheel|wheel\s+(is\s+|was\s+)?(loose|wobbl\w*)|wobbl\w*\b.{0,25}\bwheel"
    r"|lug\s+nuts?\b.{0,25}\b(loose|missing|sheared|snapped|broke)|(loose|missing|sheared)\b.{0,20}\blug\s+nuts?"
    r"|(wheel\s+)?studs?\s+(sheared|snapped|broke|broken)"
    r"|(ball\s+joint|tie\s+rod|control\s+arm|steering\s+(rack|shaft|column)|coil\s+spring|leaf\s+spring"
    r"|strut|shock|sway\s+bar|wheel\s+bearing|subframe)\b.{0,30}\b(popped|separated|snapped|broke|broken|failed"
    r"|let\s+go|came\s+apart|came\s+off|fell\s+(off|apart))"
    r"|(spring|axle|cv\s+joint|hub)\s+(snapped|broke|broken|sheared)"
    r"|steering\s+(went|is|was|feels?|felt)\s+(loose|sloppy|dead|slack)|lost\s+(my\s+|the\s+)?steering"
    r"|can'?t\s+steer|won'?t\s+steer|steering\s+locked"
    r"|tire\s+(is\s+)?rubbing\b.{0,25}\b(fender|strut|frame|body)"
    r")",
    _I,
)
_FIRE_OR_FUEL = re.compile(
    r"\b("
    r"(smell|smells?|smelling|smelt|whiff|odor|odour|reek\w*)\b.{0,30}\b(gas|gasoline|petrol|fuel|diesel)"
    r"|(gas|gasoline|petrol|fuel|diesel)\b.{0,20}\b(smell|smells|odor|odour|fumes)"
    r"|fuel\s+(leak|leaking|dripping|spraying|pouring|pooling)|(leak|leaking|dripping|spraying|pooling)\b.{0,20}\bfuel"
    r"|smoke\b.{0,30}\b(hood|bonnet|engine|dash|wheel\s+well|car|truck|van|suv|tailpipe\s+and)"
    r"|(hood|bonnet|engine|dash)\b.{0,20}\b(is\s+)?smoking"
    r"|on\s+fire|caught\s+fire|flames?\b|burning\s+smell|smells?\s+like\s+(it'?s\s+)?burning|electrical\s+burning"
    r"|batter\w*\b.{0,30}\b(swollen|swelling|bulging|hissing|leaking\s+acid|melted|smoking|venting)"
    r"|(temp|temps|temperature|gauge)\b.{0,30}\b(in\s+the\s+red|red\s+zone|pegged|maxed)"
    r"|overheat\w*\b.{0,25}\bsteam|steam\w*\b.{0,25}\b(hood|bonnet|engine|out)"
    r"|coolant\s+boil\w*|boiling\s+over"
    r")",
    _I,
)
_SPEC_QUESTION = re.compile(
    r"\b("
    r"torque|ft\s*-?\s*lbs?\b|foot\s*-?\s*pounds?|lb\s*-?\s*ft\b|newton\s*-?\s*met(er|re)s?|nm\b|in\s*-?\s*lbs?\b"
    r"|how\s+tight|tighten\w*\s+(spec\w*|torque|sequence|order|pattern)"
    r"|how\s+(much|many)\b.{0,30}\b(oil|coolant|fluid|quarts?|qts?|liters?|litres?|grease|refrigerant|gear\s+lube)"
    r"|(fill|fluid|oil|coolant|refrigerant)\s+capacit\w*|capacit\w*\b"
    r"|(spark\s+)?plug\s+gap|clearance\s+spec\w*|valve\s+clearance|end\s*play"
    r"|(tire|tyre)\s+pressure|psi\b|cold\s+inflation"
    r"|bleed\w*\s+(order|sequence)|(minimum|discard|machining)\s+thickness|runout\s+spec\w*"
    r"|(service|change|replacement)\s+interval|how\s+often\b.{0,30}\b(change|replace|rotate|flush|service|grease)"
    r"|what\s+(spec|specs|specification)\b"
    r")",
    _I,
)
_DIAGNOSIS = re.compile(
    r"\b("
    r"diagnos\w*"
    r"|what'?s\s+wrong\s+with|what\s+(could|might)\s+(be\s+)?(wrong|causing|the\s+problem)"
    r"|could\s+(it|this|that)\s+be|is\s+(it|this|that)\s+(the\s+|my\s+)?\w+\s+(going\s+bad|failing|bad|shot|dead|toast)"
    r"|do\s+i\s+need\s+(a\s+new|new|to\s+replace)"
    r"|why\s+(does|is|was|did|won'?t|doesn'?t|wont)\s+my\b"
    r"|(code|dtc)\s+p[0-3]\d{3}|\bp0\d{3}\b"
    r"|what\s+does\s+(the\s+)?(check\s+engine|that\s+(noise|sound)|this\s+(noise|sound))"
    r")",
    _I,
)

_TREAD_32NDS = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*32", _I)
_MM_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", _I)
_TREAD_WORD = re.compile(r"\b(tread|tyre|tire)s?\b", _I)
_PAD_WORD = re.compile(r"\b(pads?|friction\s+material|brake\s+shoes?|linings?)\b", _I)


@dataclass(frozen=True)
class Vehicle:
    """The customer's car. The gate on fitment and on every service figure we may quote."""

    year: int | None = None
    make: str | None = None
    model: str | None = None

    def __bool__(self) -> bool:
        return any((self.year, self.make, self.model))

    @property
    def label(self) -> str:
        return " ".join(str(p) for p in (self.year, self.make, self.model) if p)

    def filled_from(self, known: Vehicle | None) -> Vehicle:
        """Fields this message didn't state fall back to what earlier turns established."""
        if known is None:
            return self
        return Vehicle(
            year=self.year if self.year is not None else known.year,
            make=self.make or known.make,
            model=self.model or known.model,
        )


@dataclass(frozen=True)
class Measurement:
    """A wear measurement the customer gave, next to the minimum it is judged against."""

    part: str
    value: float
    unit: str
    minimum: float

    @property
    def summary(self) -> str:
        return f"{self.part} at {self._with_unit(self.value)} (minimum {self._with_unit(self.minimum)})"

    def _with_unit(self, value: float) -> str:
        # A fraction reads as "1/32 in", a plain unit as "2 mm".
        return f"{value:g}{self.unit}" if self.unit.startswith("/") else f"{value:g} {self.unit}"


@dataclass(frozen=True)
class Detection:
    flags: tuple[str, ...]
    vehicle: Vehicle | None = None
    measurement: Measurement | None = None

    @property
    def critical(self) -> bool:
        return bool(CRITICAL_FLAGS.intersection(self.flags))

    @property
    def needs_review(self) -> bool:
        return bool(REVIEW_FLAGS.intersection(self.flags))


def detect(message: str, known_vehicle: Vehicle | None = None) -> Detection:
    """Flag a customer message. `known_vehicle` carries the car from earlier turns."""
    vehicle = _vehicle(message).filled_from(known_vehicle) or None
    measurement = _below_minimum(message)

    flags: list[str] = []
    if _BRAKE_FAILURE.search(message):
        flags.append("brake_failure")
    if _LOSS_OF_CONTROL.search(message):
        flags.append("loss_of_control")
    if _FIRE_OR_FUEL.search(message):
        flags.append("fire_or_fuel")
    if measurement is not None:
        flags.append("below_minimum")
    if _SPEC_QUESTION.search(message):
        flags.append("spec_question")
    if _DIAGNOSIS.search(message):
        flags.append("diagnosis_request")
    return Detection(flags=tuple(flags), vehicle=vehicle or None, measurement=measurement)


def _vehicle(text: str) -> Vehicle:
    match = _VEHICLE.search(text)
    if match is None:
        year = _YEAR.search(text)
        return Vehicle(year=int(year.group(1))) if year else Vehicle()
    model = (match.group("model") or "").strip(" .,;:!?")
    if model.lower() in _NOT_A_MODEL or re.fullmatch(_MAKES, model, _I):
        model = ""
    raw_year = match.group("y1") or match.group("y2")
    year = int(raw_year) if raw_year else None
    if year is None and (loose := _YEAR.search(text)):
        year = int(loose.group(1))
    return Vehicle(year=year, make=match.group("make").lower(), model=model.lower() or None)


def _below_minimum(text: str) -> Measurement | None:
    """A measurement the customer gave that is at or under a legal / replace-now minimum.

    Tread in 32nds needs no context word: "2/32" is a tread depth and nothing else in a parts chat.
    A figure in mm is only read as wear if the words around it say which part it belongs to.
    """
    for match in _TREAD_32NDS.finditer(text):
        value = float(match.group(1))
        if value <= TREAD_MIN_32NDS:
            return Measurement("tire tread", value, "/32 in", TREAD_MIN_32NDS)
    for match in _MM_VALUE.finditer(text):
        value = float(match.group(1))
        window = text[max(0, match.start() - 40) : match.end() + 40]
        if _TREAD_WORD.search(window) and value <= TREAD_MIN_MM:
            return Measurement("tire tread", value, "mm", TREAD_MIN_MM)
        if _PAD_WORD.search(window) and value <= PAD_MIN_MM:
            return Measurement("brake pad friction material", value, "mm", PAD_MIN_MM)
    return None


def with_flags(detection: Detection, added: list[str]) -> Detection:
    """The same detection plus flags another layer found. Flags are only ever added."""
    return replace(detection, flags=(*detection.flags, *(f for f in added if f not in detection.flags)))
