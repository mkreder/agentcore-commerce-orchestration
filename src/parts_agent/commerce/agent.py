"""03 Commerce agent: the only agent that can act on the store.

The advisor talks to the customer; when the customer wants to buy, it delegates to this agent as a
tool (agents as tools). The advisor decides *when* to buy, which is conversation. Code decides
*whether* buying is allowed, and there are two gates:

  - the safety gate: the tools refuse to add once either detection layer flags a critical turn, or
    once the monitor has escalated the session;
  - the fitment gate: the tools refuse to add a part the fitment table rules out for the car the
    customer told us about. No prompt, no judgement, no "just add it anyway".

None of them can set a price; the discount allow-list and the quantity cap live in rules.py.

Built with the Strands harness, narrowed like the advisor: no shell, files, web or memory. Each request
starts from an empty conversation, because the cart lives in code (CommerceState), not in the transcript.
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote, urlparse

from strands import Agent, tool
from strands_harness import create_harness

from .. import specs
from ..config import Settings
from ..detector import CRITICAL_FLAGS
from ..session import CartLine, Session
from .rules import RuleError, apply_discount, check_discount_code, check_fitment, check_quantity, format_money
from .store import Store, StoreError

FLAGS_WAIT_S = 10

COMMERCE_PROMPT = """\
You operate the cart of an online auto parts store. You never talk to the customer. The store's
shopping assistant sends you one request at a time, like "add 1 of variant gid://... to the cart" or
"apply code WELCOME10 and get the checkout link".

- Do exactly what the request asks, with your tools. Look a product up with get_product when you need a
  variant id. If a request is ambiguous (several sizes or sides and none named), change nothing and
  list the options with their variant ids.
- Never invent prices, totals, stock, fitment or links: report only what the tools returned.
- If a tool returns an error, don't work around it and don't retry it with a different variant to get
  past it. Report the error as given, including anything it says about fitment.
- Reply in one to three plain sentences for the assistant: what changed, the cart lines and the total,
  the checkout link if you got one, or the error.
"""


def build_commerce_agent(settings: Settings, store: Store, session: Session) -> Agent:
    state = session.commerce

    async def add_blocked(variant_id: str, quantity: int) -> str | None:
        current = state.lines[variant_id].quantity if variant_id in state.lines else 0
        if quantity <= current:
            return None  # removing items is always allowed
        # The classifier runs in parallel with the advisor: wait for its flags before adding anything.
        # If it hangs, the regex flags already in turn_flags still apply.
        try:
            await asyncio.wait_for(session.shared.flags_settled.wait(), timeout=FLAGS_WAIT_S)
        except asyncio.TimeoutError:
            pass
        if CRITICAL_FLAGS.intersection(session.shared.turn_flags):
            return (
                "Adding items is paused: this message describes a car that isn't safe to drive. "
                "Nothing was added. The customer needs the car towed and looked at before any part matters."
            )
        if session.monitor.escalated:
            return (
                "This conversation was escalated because the car isn't safe to drive, so adding items is "
                "paused for this session. The customer can still remove items or check out what's already in the cart."
            )
        return None

    @tool
    async def get_product(product_id: str) -> dict:
        """Get one product with its variants (ids, prices in minor units, availability).

        Args:
            product_id: A product id, variant id or handle.
        """
        return await store.get_product(product_id) or {"error": "not found"}

    @tool
    async def update_cart(variant_id: str, quantity: int) -> dict:
        """Set the quantity of one variant in the customer's cart. Quantity 0 removes it.

        Args:
            variant_id: The variant id.
            quantity: The total quantity the customer wants of this variant.
        """
        if reason := await add_blocked(variant_id, quantity):
            return {"error": reason}
        fitment = None
        try:
            check_quantity(quantity, settings.max_qty_per_line)
            product = await store.get_product(variant_id)
            variant = next((v for v in (product or {}).get("variants", []) if v["id"] == variant_id), None)
            if variant is None:
                return {"error": "Unknown variant id. Look the product up first."}
            if quantity and variant.get("available") is False:
                return {"error": f"{product['title']} ({variant['title']}) is out of stock."}
            if quantity > current_quantity(state, variant_id):
                # Checked on the way up only: taking items out never needs to fit.
                fitment = check_fitment(
                    product["handle"], product["title"], session.shared.vehicle, settings.enforce_fitment
                )

            lines = dict(state.lines)
            if quantity == 0:
                lines.pop(variant_id, None)
            else:
                lines[variant_id] = CartLine(variant_id, f"{product['title']} ({variant['title']})", quantity, variant["price"] or 0)
            cart = await store.write_cart(state.cart_id, list(lines.values()))
        except RuleError as e:
            return {"error": str(e)}
        except StoreError as e:
            state.tool_errors.append(str(e))
            return {"error": f"The store rejected the change: {e}"}

        state.lines, state.cart_id, state.continue_url = lines, cart["id"], cart["continue_url"]
        state.last_cart_action = {"variant_id": variant_id, "quantity": quantity, "fitment": fitment}
        view = _cart_view(state, cart["total"], cart["currency"], settings)
        if note := _fitment_note(fitment, session):
            view["fitment"] = note
        return view

    @tool
    def view_cart() -> dict:
        """Show what's in the customer's cart and the total."""
        total = sum(l.unit_price * l.quantity for l in state.lines.values())
        return _cart_view(state, total, store.currency, settings)

    @tool
    def apply_discount_code(code: str) -> dict:
        """Apply a discount code the customer gave. Only the store's existing codes work.

        Args:
            code: The code exactly as the customer typed it.
        """
        try:
            state.discount_code, percent = check_discount_code(code, settings.discount_codes)
        except RuleError as e:
            return {"error": str(e)}
        return {"applied": state.discount_code, "percent_off": percent}

    @tool
    def checkout_link() -> dict:
        """The link where the customer reviews the cart and pays, on the store's own checkout."""
        if not state.continue_url:
            return {"error": "The cart is empty."}
        return {"url": _with_discount(state.continue_url, state.discount_code)}

    return create_harness(
        model=settings.commerce_model,
        system_prompt=COMMERCE_PROMPT,
        tools=[get_product, update_cart, view_cart, apply_discount_code, checkout_link],
        builtin_tools=[],
        builtin_plugins=[],
        background_tasks=False,
        context_manager=False,
        skills=False,
        memory=False,
        session=False,
        callback_handler=None,
    )


def as_tool(commerce: Agent, session: Session):
    """The commerce agent, exposed to the advisor as one tool."""
    lock = asyncio.Lock()  # one store request at a time, even if the advisor calls tools in parallel

    @tool
    async def commerce_agent(request: str) -> str:
        """Ask the store's commerce agent to act on the cart: add, change or remove items, show the cart,
        apply a discount code, or get the checkout link. It checks fitment against the customer's vehicle
        before adding anything, and replies with what changed and the total, or with the store's error.

        Args:
            request: Exactly what to do, with variant ids and quantities when you have them,
                e.g. "Set gid://shopify/ProductVariant/123 to quantity 1, then show the cart."
        """
        async with lock:
            state = session.commerce
            state.turn_requests.append(request)
            commerce.messages.clear()  # each request stands alone; the cart lives in code
            result = await commerce.invoke_async(request)
            for key, value in result.metrics.accumulated_usage.items():
                if isinstance(value, int):
                    state.turn_usage[key] = state.turn_usage.get(key, 0) + value
            return str(result).strip()

    return commerce_agent


def current_quantity(state, variant_id: str) -> int:
    return state.lines[variant_id].quantity if variant_id in state.lines else 0


def _fitment_note(status: str | None, session: Session) -> str | None:
    """What the assistant should tell the customer about fitment, when code couldn't confirm it."""
    if status == specs.UNKNOWN:
        if session.shared.vehicle and session.shared.vehicle.make:
            return f"Fitment not confirmed for a {session.shared.vehicle.label}: ask the customer for the model year."
        return "Fitment not checked: no vehicle on file. Ask the customer for the year, make, model and engine."
    if status == specs.UNIVERSAL:
        return "Listed as a universal fit, so no vehicle check was needed."
    return None


def _cart_view(state, total: int | None, currency: str, settings: Settings) -> dict:
    view = {
        "lines": [{"title": l.title, "quantity": l.quantity, "unit_price": format_money(l.unit_price, currency)} for l in state.lines.values()],
        "total": format_money(total, currency),
    }
    if state.discount_code and total is not None:
        pct = settings.discount_codes[state.discount_code]
        view["discount"] = f"{state.discount_code} (-{pct}%)"
        view["total_after_discount"] = format_money(apply_discount(total, pct), currency)
    return view


def _with_discount(continue_url: str, code: str | None) -> str:
    """Shopify's shareable discount link: /discount/{code}?redirect={path}."""
    if not code:
        return continue_url
    u = urlparse(continue_url)
    path = u.path + (f"?{u.query}" if u.query else "")
    return f"{u.scheme}://{u.netloc}/discount/{quote(code)}?redirect={quote(path, safe='')}"
