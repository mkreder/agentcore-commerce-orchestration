"""Amazon Bedrock AgentCore Runtime entrypoint.

AgentCore runs this container and gives every session its own microVM. An async-generator entrypoint
is served as Server-Sent Events, so the widget gets text as it is generated.

    payload: {"prompt": "Do you have ceramic front brake pads for a 2018 Honda Civic?"}
    session: the runtime session id (header X-Amzn-Bedrock-AgentCore-Runtime-Session-Id)
"""

from __future__ import annotations

import logging

from bedrock_agentcore.runtime import BedrockAgentCoreApp, RequestContext

from .config import Settings
from .pipeline import Orchestrator

logging.basicConfig(level=logging.INFO)

app = BedrockAgentCoreApp()
orchestrator = Orchestrator(Settings.from_env())


@app.entrypoint
async def invoke(payload: dict, context: RequestContext):
    prompt = (payload or {}).get("prompt")
    if not prompt:
        yield {"type": "error", "message": 'Send {"prompt": "..."}'}
        return
    session_id = context.session_id or payload.get("session_id") or "local-session"
    async for event in orchestrator.run_turn(session_id, prompt):
        yield event


if __name__ == "__main__":
    app.run()
