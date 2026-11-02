"""The store: a local mock, or any Shopify store over UCP MCP.

Shopify exposes catalog, cart and checkout to agents at https://{shop}/api/ucp/mcp. We do not hand
that whole tool list to a model. The advisor gets read-only tools (advisor.py) and the commerce agent
gets the cart tools (commerce/agent.py); both call the few UCP tools we allow-list below. Checkout
(create_checkout) is phase 2: today the customer finishes on the store's own cart page via continue_url.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import uuid
from typing import Protocol

import httpx

from ..config import DATA_DIR, Settings
from ..session import CartLine

# The only UCP tools this agent may call. Everything else Shopify exposes stays invisible.
UCP_ALLOWED_TOOLS = frozenset({"search_catalog", "get_product", "create_cart", "update_cart", "get_cart"})


class StoreError(RuntimeError):
    pass


class Store(Protocol):
    currency: str

    async def search(self, query: str, limit: int = 5) -> list[dict]: ...
    async def get_product(self, product_id: str) -> dict | None: ...
    async def write_cart(self, cart_id: str | None, lines: list[CartLine]) -> dict: ...


def make_store(settings: Settings) -> Store:
    if settings.store_mode == "shopify":
        if not settings.shop_domain:
            raise ValueError("STORE_MODE=shopify needs SHOP_DOMAIN, e.g. your-store.myshopify.com")
        return ShopifyUcpStore(settings.shop_domain, settings.ucp_agent_profile)
    return MockStore()


class MockStore:
    """In-memory catalog from data/catalog.json. Same shapes as the Shopify store."""

    def __init__(self) -> None:
        with open(DATA_DIR / "catalog.json") as f:
            data = json.load(f)
        self.currency = data["currency"]
        self._products = data["products"]
        self._carts: dict[str, list[CartLine]] = {}

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        terms = [t for t in query.lower().replace("-", " ").split() if len(t) > 2]

        def score(p: dict) -> int:
            blob = " ".join([p["title"], p["description"], " ".join(p["tags"])]).lower().replace("-", " ")
            return sum(t.rstrip("s") in blob for t in terms)

        ranked = sorted(self._products, key=score, reverse=True)
        return [_mock_summary(p, self.currency) for p in ranked if score(p) > 0][:limit]

    async def get_product(self, product_id: str) -> dict | None:
        for p in self._products:
            if product_id in (p["id"], p["handle"]) or any(v["id"] == product_id for v in p["variants"]):
                return _mock_summary(p, self.currency)
        return None

    async def write_cart(self, cart_id: str | None, lines: list[CartLine]) -> dict:
        cart_id = cart_id or f"gid://mock/Cart/{uuid.uuid4().hex[:10]}"
        self._carts[cart_id] = list(lines)
        total = sum(line.unit_price * line.quantity for line in lines)
        return {
            "id": cart_id,
            "currency": self.currency,
            "total": total,
            "continue_url": f"https://example-parts.local/cart/c/{cart_id.rsplit('/', 1)[-1]}",
            "messages": [],
        }


def _mock_summary(p: dict, currency: str) -> dict:
    return {
        "id": p["id"],
        "handle": p["handle"],
        "title": p["title"],
        "description": p["description"],
        "currency": currency,
        "variants": [dict(v) for v in p["variants"]],
    }


class ShopifyUcpStore:
    """Shopify Storefront Catalog + Cart MCP (UCP). Anonymous access, no API key needed.

    Calls are throttled and retried: tool fan-out under load returns 429s otherwise.
    """

    def __init__(self, shop_domain: str, agent_profile: str, max_concurrency: int = 4, max_retries: int = 3):
        self.endpoint = f"https://{shop_domain}/api/ucp/mcp"
        self.agent_profile = agent_profile
        self.currency = "USD"
        self._client = httpx.AsyncClient(timeout=20)
        self._limit = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._ids = itertools.count(1)

    async def _call(self, tool: str, arguments: dict) -> dict:
        if tool not in UCP_ALLOWED_TOOLS:
            raise StoreError(f"tool {tool!r} is not allow-listed")
        body = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": next(self._ids),
            "params": {"name": tool, "arguments": {"meta": {"ucp-agent": {"profile": self.agent_profile}}, **arguments}},
        }
        async with self._limit:
            for attempt in range(self._max_retries + 1):
                resp = await self._client.post(self.endpoint, json=body)
                if resp.status_code != 429 or attempt == self._max_retries:
                    break
                await asyncio.sleep(float(resp.headers.get("Retry-After", 0.5 * 2**attempt)))
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise StoreError(payload["error"].get("message", str(payload["error"])))
        result = payload.get("result", {})
        if "structuredContent" in result:
            return result["structuredContent"]
        # Some servers only return text content: parse the first JSON block.
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except ValueError:
                    pass
        return result

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        data = await self._call("search_catalog", {"catalog": {"query": query, "pagination": {"limit": limit}}})
        return [self._summary(p) for p in data.get("products", [])][:limit]

    async def get_product(self, product_id: str) -> dict | None:
        data = await self._call("get_product", {"catalog": {"id": product_id}})
        product = data.get("product")
        return self._summary(product) if product else None

    async def write_cart(self, cart_id: str | None, lines: list[CartLine]) -> dict:
        cart = {"line_items": [{"quantity": l.quantity, "item": {"id": l.variant_id}} for l in lines]}
        if cart_id:
            # update_cart has PUT semantics: send the full desired state every time.
            data = await self._call("update_cart", {"id": cart_id, "cart": cart})
        else:
            data = await self._call("create_cart", {"cart": cart})
        c = data.get("cart", data)
        errors = [m for m in c.get("messages", []) if m.get("type") == "error" or m.get("severity") == "unrecoverable"]
        if errors:
            raise StoreError("; ".join(m.get("content") or m.get("code", "cart error") for m in errors))
        total = next((t["amount"] for t in c.get("totals", []) if t.get("type") == "total"), None)
        self.currency = c.get("currency", self.currency)
        return {
            "id": c["id"],
            "currency": self.currency,
            "total": total,
            "continue_url": c.get("continue_url"),
            "messages": c.get("messages", []),
        }

    def _summary(self, p: dict) -> dict:
        variants = []
        for v in p.get("variants", []):
            price = v.get("price") or {}
            variants.append({
                "id": v.get("id"),
                "title": v.get("title"),
                "price": price.get("amount") if isinstance(price, dict) else price,
                "available": _available(v.get("availability")),
            })
        return {
            "id": p.get("id"),
            "handle": p.get("handle"),
            "title": p.get("title"),
            "description": _text(p.get("description"))[:600],
            "url": p.get("url"),
            "currency": ((p.get("price_range") or {}).get("min") or {}).get("currency", self.currency),
            "variants": variants,
        }


def _text(description) -> str:
    """UCP descriptions are objects like {"html": "..."} or {"plain": "..."}."""
    if isinstance(description, dict):
        description = description.get("plain") or description.get("html") or ""
    return re.sub(r"<[^>]+>", " ", description or "").strip()


def _available(availability) -> bool | None:
    if isinstance(availability, bool) or availability is None:
        return availability
    if isinstance(availability, dict):
        return availability.get("available", availability.get("status") in ("in_stock", "available"))
    return str(availability).lower() in ("in_stock", "available", "true")
