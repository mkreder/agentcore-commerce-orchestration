"""Settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("PARTS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

# A hosted test profile from Shopify. Host your own agent profile before going to production:
# https://shopify.dev/docs/agents/profiles
DEFAULT_UCP_PROFILE = "https://shopify.dev/ucp/agent-profiles/2026-08-25/valid-with-capabilities.json"

# Widely enabled on Bedrock. Swap in newer models with ADVISOR_MODEL / COMMERCE_MODEL / MONITOR_MODEL.
# The commerce agent runs short, well-specified store requests, and the monitor reads a short payload
# on a minority of turns: a small, fast model fits both.
DEFAULT_ADVISOR_MODEL = "bedrock/global.anthropic.claude-sonnet-4-6"
DEFAULT_COMMERCE_MODEL = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_MONITOR_MODEL = "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
# The classifier runs on most turns and only returns flags, so a small, fast model fits. Picking a
# different model family than the monitor keeps the two from sharing blind spots.
DEFAULT_CLASSIFIER_MODEL = "bedrock/global.openai.gpt-5.6-luna"


@dataclass(frozen=True)
class Settings:
    store_mode: str  # "mock" or "shopify"
    shop_domain: str | None
    ucp_agent_profile: str
    advisor_model: str | None  # "provider/name" or a Bedrock id; None = Strands harness default
    commerce_model: str | None
    monitor_model: str | None  # a Bedrock model id; the "bedrock/" prefix is optional
    classifier: str  # "model" (second detection layer) or "off" (regex only)
    classifier_model: str | None  # a Bedrock model id
    classifier_timeout_s: float  # past this, the classifier is cut off and the turn goes to review
    classifier_reasoning_effort: str | None  # OpenAI models only: "none", "low", ...; None = model default
    discount_codes: dict[str, int]  # code -> percent off, the only discounts that exist
    max_qty_per_line: int
    enforce_fitment: bool  # refuse, in code, to add a part the fitment table says does not fit
    session_dir: str

    @classmethod
    def from_env(cls) -> Settings:
        shop = os.environ.get("SHOP_DOMAIN") or None
        return cls(
            store_mode=os.environ.get("STORE_MODE", "shopify" if shop else "mock"),
            shop_domain=shop,
            ucp_agent_profile=os.environ.get("UCP_AGENT_PROFILE", DEFAULT_UCP_PROFILE),
            advisor_model=os.environ.get("ADVISOR_MODEL") or DEFAULT_ADVISOR_MODEL,
            commerce_model=os.environ.get("COMMERCE_MODEL") or DEFAULT_COMMERCE_MODEL,
            monitor_model=os.environ.get("MONITOR_MODEL") or DEFAULT_MONITOR_MODEL,
            classifier=os.environ.get("CLASSIFIER", "model"),
            classifier_model=os.environ.get("CLASSIFIER_MODEL") or DEFAULT_CLASSIFIER_MODEL,
            classifier_timeout_s=float(os.environ.get("CLASSIFIER_TIMEOUT_S", "4")),
            classifier_reasoning_effort=os.environ.get("CLASSIFIER_REASONING_EFFORT", "none") or None,
            discount_codes=_parse_codes(os.environ.get("DISCOUNT_CODES", "WELCOME10:10")),
            max_qty_per_line=int(os.environ.get("MAX_QTY_PER_LINE", "6")),
            enforce_fitment=os.environ.get("ENFORCE_FITMENT", "true").lower() not in ("0", "false", "off"),
            session_dir=os.environ.get("SESSION_DIR", "./.agent/sessions"),
        )


def _parse_codes(raw: str) -> dict[str, int]:
    codes = {}
    for item in filter(None, (part.strip() for part in raw.split(","))):
        code, _, pct = item.partition(":")
        codes[code.upper()] = int(pct or 0)
    return codes
