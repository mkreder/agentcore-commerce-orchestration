"""Service data and fitment: SHARED memory, read by both agents.

A torque figure, a capacity, a clearance or an interval can only come from here. If there is no band
for the customer's model year, the figure is None, and no figure may appear in the answer.

Fitment lives here too, as data. The *refusal* that uses it is in commerce/rules.py, so that the
part that says no is code on the write path, not a sentence in a prompt.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .config import DATA_DIR
from .detector import Vehicle

# Everyday names customers use, mapped to the make in the fitment table.
MAKE_ALIASES = {
    "chevy": "chevrolet",
    "vw": "volkswagen",
    "mercedes": "mercedes-benz",
    "benz": "mercedes-benz",
    "range rover": "land rover",
}

FITS, DOES_NOT_FIT, UNIVERSAL, UNKNOWN = "fits", "does_not_fit", "universal", "unknown"


@lru_cache(maxsize=1)
def _specs() -> dict[str, dict]:
    with open(DATA_DIR / "specs.json") as f:
        return json.load(f)["specs"]


def get_specs(handle: str) -> dict | None:
    spec = _specs().get(handle)
    if spec is None:
        return None
    return {k: v for k, v in spec.items() if k != "keywords"}


def mentioned_products(*texts: str) -> list[str]:
    """Handles of products with service data that a message or draft talks about."""
    blob = " ".join(texts).lower()
    return [h for h, spec in _specs().items() if any(k in blob for k in spec["keywords"])]


def spec_for_year(handle: str, year: int | None) -> str | None:
    spec = _specs().get(handle)
    if spec is None or year is None:
        return None
    for band in spec["bands"]:
        if band["year_min"] <= year <= band["year_max"]:
            return band["spec"]
    return None


def allowed_specs(handles: list[str], year: int | None) -> list[str]:
    """Figures an answer may quote: the band for the customer's model year, or every band when the
    year is unknown."""
    allowed = []
    for handle in handles:
        spec = _specs().get(handle)
        if spec is None:
            continue
        if year is None:
            allowed += [band["spec"] for band in spec["bands"]]
        elif found := spec_for_year(handle, year):
            allowed.append(found)
    return allowed


def fitment(handle: str, vehicle: Vehicle | None) -> str:
    """Does this part fit that car? FITS, DOES_NOT_FIT, UNIVERSAL or UNKNOWN.

    UNKNOWN means we cannot tell from what the customer has told us: no vehicle, no make, or a make
    and model that match but no model year. Only DOES_NOT_FIT ever blocks a purchase.
    """
    spec = _specs().get(handle)
    if spec is None:
        return UNKNOWN
    listed = spec.get("fits") or []
    if not listed:
        return UNIVERSAL
    if vehicle is None or not vehicle.make:
        return UNKNOWN

    make = MAKE_ALIASES.get(vehicle.make.lower(), vehicle.make.lower())
    candidates = [row for row in listed if row["make"].lower() == make]
    if not candidates:
        return DOES_NOT_FIT
    if vehicle.model:
        by_model = [row for row in candidates if row.get("model", "").lower() == vehicle.model.lower()]
        if not by_model:
            return DOES_NOT_FIT
        candidates = by_model
    if vehicle.year is None:
        return UNKNOWN
    if any(row["year_min"] <= vehicle.year <= row["year_max"] for row in candidates):
        return FITS
    return DOES_NOT_FIT


def listed_years(handle: str) -> str:
    """How the fitment table reads, for an error message a customer will see."""
    rows = (_specs().get(handle) or {}).get("fits") or []
    return "; ".join(
        " ".join(filter(None, [row["make"].title(), row.get("model", "").title(), f"{row['year_min']}-{row['year_max']}"]))
        for row in rows
    )
