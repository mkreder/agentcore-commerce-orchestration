"""Chat with the deployed AgentCore Runtime and print the SSE stream.

    python scripts/invoke.py <runtime-arn> "Do you have front brake pads for a 2018 Honda Civic?"
    python scripts/invoke.py <runtime-arn> "Add a set to my cart" --session <same id as before>
"""

from __future__ import annotations

import argparse
import json
import uuid

import boto3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_arn")
    parser.add_argument("prompt")
    parser.add_argument("--session", default=None, help="reuse to continue a conversation (min 33 chars)")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    session_id = args.session or f"cli-{uuid.uuid4()}"
    client = boto3.client("bedrock-agentcore", region_name=args.region or args.runtime_arn.split(":")[3])
    response = client.invoke_agent_runtime(
        agentRuntimeArn=args.runtime_arn,
        runtimeSessionId=session_id,
        payload=json.dumps({"prompt": args.prompt}).encode(),
    )
    for line in response["response"].iter_lines():
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[len(b"data: "):])
        if "type" not in event:  # an error raised inside the runtime, reported by the AgentCore SDK
            print(f"runtime error ({event.get('error_type', 'unknown')}): {event.get('error', event)}")
        elif event["type"] == "text":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "replace":
            print(f"\n[monitor rewrote the answer]\n{event['text']}", end="")
        elif event["type"] == "done":
            print(f"\n\ntrace: {json.dumps(event['trace'])}")
        elif event["type"] == "error":
            print(f"error: {event['message']}")
    print(f"\nsession: {session_id}")


if __name__ == "__main__":
    main()
