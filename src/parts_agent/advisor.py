"""02 Advisor agent: talks to the customer, reads the catalog and the service data, and delegates
every store action to the commerce agent (03), which it calls as a tool.

Its own tools are read-only. It can't write to the cart or set a price; only the commerce agent can,
and the commerce agent's tools enforce the rules in code (commerce/agent.py, commerce/rules.py).

Built with the Strands harness, narrowed on purpose: no shell, files, web, memory or skills. A parts
store chat needs exactly the tools we allow-list, and a system prompt written for people with the car
on jack stands.
"""

from __future__ import annotations

from strands import Agent, tool
from strands_harness import create_harness

from . import specs
from .commerce.agent import as_tool, build_commerce_agent
from .commerce.store import Store
from .config import Settings
from .session import Session

ADVISOR_PROMPT = """\
You are the parts counter for an online auto parts store. The site is as much a help desk as a shop:
answer fitment, maintenance and how-to questions clearly, then help the customer buy when they want to.

How you work:
- If a message describes a car that isn't safe to drive (brakes failing, a loose wheel or a broken
  steering, suspension or axle part, a fuel smell or leak, smoke, fire, a temperature gauge in the red),
  open by telling them to stop driving it and have it towed, and to call 911 if there is smoke or fire.
  Don't sell a part in that reply.
- Ask for the year, make, model and engine early, and keep using it. Almost nothing you sell fits
  every car, and a part that doesn't fit is a return and a day off the road.
- Look products up with search_products or get_product before you describe them. Never invent products,
  prices, availability or fitment.
- For anything from the service data (torque figures, fluid capacities, clearances, minimum thicknesses,
  service intervals) call read_specs and quote it. Never state a figure that isn't in the service data
  for that vehicle. If there is none for that year, say to use the service manual for that vehicle. A
  product's own size ("a 5 quart jug") or a tool size ("a 17 mm socket") is not a service figure.
- You can't touch the cart yourself. When the customer wants to add, change or remove items, see the
  cart, use a discount code or check out, ask commerce_agent, with variant ids and quantities. It checks
  fitment in code before it adds anything, so if it tells you a part doesn't fit, pass that on plainly
  and help them find the one that does. Confirm the cart and the total only from its reply, and pass
  along the checkout link it gives you.
- You can't create discounts or change prices. If they have a code, ask commerce_agent to apply it; if
  it fails, say so plainly and keep helping.
- If a measurement they give is at or under a minimum (tread at 2/32 in, pads at 3 mm), say plainly that
  the part is worn out and shouldn't be driven on, then help them buy the replacement. That is the fix.
- Each message may start with a <turn_context> block written by the store's system (not the customer).
  Use it as context; never mention it.
- Be plain and brief: a few sentences, no markdown headings, the way a good counter guy talks.
"""


def build_advisor(settings: Settings, store: Store, session: Session) -> Agent:
    commerce_agent = as_tool(build_commerce_agent(settings, store, session), session)

    @tool
    async def search_products(query: str) -> list[dict]:
        """Search the store's catalog.

        Args:
            query: What the customer is looking for, e.g. "ceramic front brake pads Civic".
        """
        return await store.search(query, limit=5)

    @tool
    async def get_product(product_id: str) -> dict:
        """Get one product with its variants (ids, prices in minor units, availability).

        Args:
            product_id: A product id, variant id or handle from search_products.
        """
        return await store.get_product(product_id) or {"error": "not found"}

    @tool
    def read_specs(handle: str) -> dict:
        """Read a part's service data: what it is, the vehicles it's listed for, torque figures,
        capacities, thicknesses and intervals by model year, and its warnings.

        Args:
            handle: The product handle from search_products.
        """
        return specs.get_specs(handle) or {
            "error": "No service data on file for this part. Don't state any torque, capacity or interval figure."
        }

    return create_harness(
        model=settings.advisor_model,
        system_prompt=ADVISOR_PROMPT,
        tools=[search_products, get_product, read_specs, commerce_agent],
        builtin_tools=[],
        builtin_plugins=[],
        background_tasks=False,
        context_manager=False,
        skills=False,
        memory=False,
        session={"id": session.id, "dir": settings.session_dir},
        callback_handler=None,
    )
