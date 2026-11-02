"""04 Review gate: code decides whether a draft answer needs a second look.

Most turns don't, and skip the monitor entirely. That is where the latency and token savings are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .detector import REVIEW_FLAGS, Detection

# A service figure stated in the draft: a torque, a capacity, a pressure, a clearance or an interval.
# None of these reaches the customer unchecked, because a wrong one ruins a part or lets it come loose.
SPEC_IN_TEXT = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*-?\s*"
    r"(ft\.?\s*-?\s*lbs?\b|foot\s*-?\s*pounds?\b|lb\.?\s*-?\s*ft\b|nm\b|newton\s*-?\s*met(?:er|re)s?\b"
    r"|in\.?\s*-?\s*lbs?\b|inch\s*-?\s*pounds?\b"
    r"|psi\b|kpa\b|bar\b"
    r"|qts?\b|quarts?\b|liters?\b|litres?\b"
    r"|miles?\b|km\b|kilometer?s?\b"
    r"|mm\b)",
    re.IGNORECASE,
)
# A figure that describes the product or a tool is not a service figure. "a 5 quart jug" is what you
# buy; "3.7 quarts" is what the engine takes. "a 17 mm socket" is what you turn the bolt with;
# "80 ft-lbs" is how tight. "rated to 60,000 miles" is a warranty; "every 7,500 miles" is an interval.
PRODUCT_ATTRIBUTE = re.compile(
    r"\b(?:contains?|holds?|comes?\s+in|sold\s+(?:as|in)|packaged\s+(?:as|in))\s+(?:an?\s+|the\s+)?"
    r"\d+(?:[.,]\d+)?\s*-?\s*(?:qts?|quarts?|gal|gallons?|fl\.?\s*oz|oz|ml|liters?|litres?)\b"
    r"|\b(?:jug|bottle|container|pack|case|jar|can|box|kit)s?\s+(?:is|are|holds?|contains?|has|have|comes?\s+with)\s+"
    r"(?:an?\s+)?\d+(?:[.,]\d+)?\s*-?\s*(?:qts?|quarts?|gal|gallons?|fl\.?\s*oz|oz|ml|liters?|litres?)\b"
    r"|\b\d+(?:[.,]\d+)?\s*-?\s*(?:qts?|quarts?|gal|gallons?|fl\.?\s*oz|oz|ml|liters?|litres?)\s*-?\s*"
    r"(?:jug|bottle|container|pack|case|jar|can|box|kit|size)\b"
    r"|\(\s*\d+(?:[.,]\d+)?\s*(?:qts?|quarts?|ml|fl\.?\s*oz|liters?|litres?|mm|in|inch(?:es)?)\s*\)"
    r"|\b\d+(?:[.,]\d+)?\s*-?\s*(?:mm|in\.?|inch(?:es)?|\")\s*-?\s*(?:[\w-]+\s+)?"
    r"(?:socket|wrench|spanner|hex|allen|bit|driver|drive|extension|blades?|rotors?|discs?|disks?|drums?|rims?"
    r"|wheels?|tires?|tyres?|sway\s+bar|bolts?|studs?|threads?|diameter|long|wide|tall|pipe|hose)\b"
    r"|\b(?:rotor|disc|disk|blade|blades|rim|wheel|drum|bolt\s+pattern|diameter|width|length|thread)s?\s+"
    r"(?:diameter\s+)?(?:is\s+|are\s+|of\s+|measures?\s+)?\d+(?:[.,]\d+)?\s*-?\s*(?:mm|in\.?|inch(?:es)?|\")\b"
    r"|\b(?:rated|warrantied|backed|good|guaranteed|covered)\s+(?:to|for|up\s+to)\s+[\d,]+(?:\.\d+)?\s*-?\s*"
    r"(?:miles?|km|kilometers?)\b"
    r"|\b[\d,]+(?:\.\d+)?\s*-?\s*(?:miles?|km|kilometers?)\s*-?\s*(?:warrant\w*|rating|rated|guarantee\w*)\b"
    r"|\b\d+\s*-?\s*(?:pack|piece|pcs?|count|ct|sets?)\b|\bset\s+of\s+\d+\b"
    r"|\$\s?\d+(?:[.,]\d+)?"
    r"|\b\d+w\s*-?\s*\d+\b|\bdot\s*\d\b|\bsae\s*\d+\b|\bapi\s+s[a-z]\b"
    r"|\b\d+\s*x\s*\d+(?:[.,]\d+)?\b",
    re.IGNORECASE,
)
# English and Spanish: an answer that already tells them to stop driving must not be replaced.
STOP_DRIVING_IN_TEXT = re.compile(
    r"\b(stop\s+driving|don'?t\s+drive|do\s+not\s+drive|not\s+safe\s+to\s+drive|unsafe\s+to\s+drive"
    r"|have\s+it\s+towed|get\s+it\s+towed|call\s+(a\s+)?tow|tow\s+truck|towed\s+to|flat\s*bed"
    r"|roadside\s+assistance|call\s+911|fire\s+department"
    r"|no\s+lo\s+(maneje|manejes|conduzca|conduzcas|mueva)|no\s+(maneje|manejes|conduzca|conduzcas)\s+"
    r"|no\s+siga\s+manejando|llam[aeá]\s+(a\s+)?(una\s+)?gr[uú]a|pida\s+(una\s+)?gr[uú]a|remolque"
    r"|bomberos|llame\s+al\s+911)",
    re.IGNORECASE,
)
# A draft that already warns about a worn-out part. If it does, the warning isn't prepended twice.
WEAR_WARNING_IN_TEXT = re.compile(
    r"\b(below\s+(the\s+)?(legal\s+|minimum|discard)|at\s+the\s+legal\s+minimum|worn\s+out|worn\s+past"
    r"|replace\s+(it|them|those|the\s+\w+)\s+(before|now)|needs?\s+replacing\s+now|shouldn'?t\s+be\s+driven"
    r"|don'?t\s+drive|stop\s+driving|not\s+safe\s+to\s+drive)",
    re.IGNORECASE,
)

_UNIT_FORMS = {
    "ftlb": "ft-lb", "ftlbs": "ft-lb", "footpound": "ft-lb", "footpounds": "ft-lb", "lbft": "ft-lb",
    "nm": "nm", "newtonmeter": "nm", "newtonmeters": "nm", "newtonmetre": "nm", "newtonmetres": "nm",
    "inlb": "in-lb", "inlbs": "in-lb", "inchpound": "in-lb", "inchpounds": "in-lb",
    "qt": "qt", "qts": "qt", "quart": "qt", "quarts": "qt",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "mile": "mile", "miles": "mile", "km": "km", "kilometer": "km", "kilometers": "km",
    "psi": "psi", "kpa": "kpa", "bar": "bar", "mm": "mm",
}


def spec_mentions(text: str) -> list[str]:
    """Every service figure stated in a text, ignoring product and tool sizes, normalized:
    "100 Ft-Lbs" -> "100 ft-lb", "3.70 quarts" -> "3.7 qt"."""
    figures = []
    plain = re.sub(r"[*_`]", "", text)  # markdown emphasis: "**100 ft-lbs**" reads as "100 ft-lbs"
    for match in SPEC_IN_TEXT.finditer(PRODUCT_ATTRIBUTE.sub(" ", plain)):
        amount = match.group(1).replace(",", "")
        if "." in amount:
            amount = amount.rstrip("0").rstrip(".")
        unit = _UNIT_FORMS.get(re.sub(r"[^a-z]", "", match.group(2).lower()), match.group(2).lower())
        figures.append(f"{amount} {unit}")
    return figures


@dataclass(frozen=True)
class GateDecision:
    review: bool
    reasons: tuple[str, ...]


def stream_mode(detection: Detection) -> bool:
    """Decided before the advisor runs: can this turn stream straight to the widget?

    Turns we already know will be reviewed are buffered, because the customer must never see a draft
    the monitor is about to rewrite. Everything else streams immediately.
    """
    return not detection.needs_review


def review_gate(detection: Detection, draft: str) -> GateDecision:
    reasons: list[str] = [f"flag:{f}" for f in detection.flags if f in REVIEW_FLAGS]
    if spec_mentions(draft):
        reasons.append("draft_states_spec")
    return GateDecision(review=bool(reasons), reasons=tuple(reasons))
