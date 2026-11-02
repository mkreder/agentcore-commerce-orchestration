"""Business rules are code, not prompt.

"Give me 50% off" stops working because no discount exists outside this allow-list, and the model has
no tool that sets a price. "Just add the pads anyway" stops working because the fitment table, not the
conversation, decides whether a part can go in the cart for the car the customer told us about.
"""

from __future__ import annotations

from .. import specs


class RuleError(ValueError):
    """A business rule said no. The message is safe to show the model."""


def check_quantity(quantity: int, max_qty: int) -> int:
    if quantity < 0:
        raise RuleError("Quantity can't be negative.")
    if quantity > max_qty:
        raise RuleError(f"The store limits this item to {max_qty} per order.")
    return quantity


def check_discount_code(code: str, allowed: dict[str, int]) -> tuple[str, int]:
    normalized = code.strip().upper()
    if normalized not in allowed:
        raise RuleError(
            "That isn't a valid discount code. The assistant can't create discounts or change prices."
        )
    return normalized, allowed[normalized]


def check_fitment(handle: str, title: str, vehicle, enforce: bool = True) -> str:
    """Refuse, in code, to put a part in the cart that the fitment table rules out for this car.

    Only an explicit DOES_NOT_FIT blocks. "No vehicle on file yet" and "fits everything" both go
    through, and the caller surfaces which of the two it was, so the assistant can ask.
    """
    status = specs.fitment(handle, vehicle)
    if enforce and status == specs.DOES_NOT_FIT:
        listed = specs.listed_years(handle) or "another vehicle"
        raise RuleError(
            f"{title} doesn't fit a {vehicle.label}. The fitment table lists it for: {listed}. "
            "Nothing was added. Confirm the year, make, model and engine with the customer, or offer a part that fits."
        )
    return status


def apply_discount(total: int, percent: int) -> int:
    return total - (total * percent) // 100


def format_money(amount: int | None, currency: str) -> str:
    if amount is None:
        return "unknown"
    return f"{amount / 100:,.2f} {currency}"
