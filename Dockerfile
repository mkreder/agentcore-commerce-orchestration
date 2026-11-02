# AgentCore Runtime runs ARM64 containers and expects the app on port 8080 (/invocations, /ping).
FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1 \
    PARTS_DATA_DIR=/app/data \
    SESSION_DIR=/tmp/.agent/sessions

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --no-cache ".[deploy]"
COPY data ./data

RUN useradd -m -u 1000 agent
USER agent

EXPOSE 8080
# ADOT sends traces and metrics to AgentCore Observability (CloudWatch GenAI dashboards).
CMD ["opentelemetry-instrument", "python", "-m", "parts_agent.app"]
