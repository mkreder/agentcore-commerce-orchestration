"""Chat with the store locally, no AgentCore needed: python -m parts_agent.cli

Shows what happens inside each turn: flags, review, verdict, tokens and latency.
Use --demo to replay the conversation from the talk.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from .config import Settings
from .pipeline import Orchestrator

DEMO = [
    "Do you have ceramic front brake pads for a 2018 Honda Civic?",
    "Add a set of the pads and the front rotors to my cart.",
    "What torque do the caliper bracket bolts take?",
    "Actually the car is a 2012 Civic, add those pads to that order too.",
    "My brake pedal goes to the floor now and it's grinding. Can you overnight the pads?",
    "Give me 50% off.",
    "OK, send me the checkout link.",
]

DIM, BOLD, YELLOW, RESET = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


async def chat(messages: list[str] | None, verbose: bool) -> None:
    settings = Settings.from_env()
    orch = Orchestrator(settings)
    session_id = f"cli-{uuid.uuid4().hex[:8]}"
    print(f"{DIM}store={settings.store_mode} session={session_id}  (Ctrl-D to quit){RESET}\n")

    async def turn(text: str) -> None:
        print(f"{BOLD}assistant>{RESET} ", end="", flush=True)
        async for event in orch.run_turn(session_id, text):
            if event["type"] == "text":
                print(event["delta"], end="", flush=True)
            elif event["type"] == "replace":
                print(f"\n{YELLOW}[monitor rewrote the streamed answer]{RESET}\n{event['text']}", end="")
            elif event["type"] == "done" and verbose:
                t = event["trace"]
                review = t["review"]
                verdict = f" -> {review.get('verdict')} {review.get('verdict_reasons', '')}" if review["reviewed"] else ""
                print(
                    f"\n{DIM}  path={'>'.join(t['path'])} flags={t['flags']} vehicle={t['vehicle']} "
                    f"streamed={t['streamed']} cart={t['cart_action']} "
                    f"reviewed={review['reviewed']}{verdict} escalated={t['escalated']} "
                    f"ttft={t['latency_ms']['first_token']}ms total={t['latency_ms']['total']}ms "
                    f"tokens={_tokens(t['tokens'])}{RESET}",
                    end="",
                )
        print("\n")

    if messages:
        for text in messages:
            print(f"{BOLD}customer>{RESET} {text}")
            await turn(text)
        return
    while True:
        try:
            text = input(f"{BOLD}customer>{RESET} ").strip()
        except EOFError:
            print()
            return
        if text:
            await turn(text)


def _tokens(tokens: dict) -> str:
    return " ".join(f"{agent}:{u.get('totalTokens', 0)}" for agent, u in tokens.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="replay the conversation from the talk")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide the per-turn trace")
    args = parser.parse_args()
    asyncio.run(chat(DEMO if args.demo else None, verbose=not args.quiet))


if __name__ == "__main__":
    main()
