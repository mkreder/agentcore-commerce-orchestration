"""One AgentCore Runtime that hosts the whole turn: detector, advisor, gate and monitor.

Both agents run in the same runtime session on purpose: the handoff to the monitor is an in-process
function call, so a flagged turn costs one extra model call and no network hop.

Context flags:
  -c tracing=true   send traces to X-Ray / AgentCore Observability (enable CloudWatch Transaction Search first)

Two ways to ship the code, picked with `-c artifact=...`:
  code       (default) direct code deployment: a .zip with Linux ARM64 wheels, built locally with uv.
             No Docker needed, and updates deploy faster.
  container  the Dockerfile at the repo root, built as an ARM64 image. Needs Docker or Finch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import jsii
from aws_cdk import BundlingOptions, CfnOutput, DockerImage, ILocalBundling, Stack
from aws_cdk import aws_bedrockagentcore as agentcore
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12"

# Read at synth time, so `SHOP_DOMAIN=... cdk deploy` configures the runtime.
PASSTHROUGH_ENV = [
    "SHOP_DOMAIN",
    "UCP_AGENT_PROFILE",
    "ADVISOR_MODEL",
    "COMMERCE_MODEL",
    "MONITOR_MODEL",
    "CLASSIFIER",
    "CLASSIFIER_MODEL",
    "CLASSIFIER_TIMEOUT_S",
    "CLASSIFIER_REASONING_EFFORT",
    "DISCOUNT_CODES",
    "MAX_QTY_PER_LINE",
    "ENFORCE_FITMENT",
]

MAIN_PY = '''\
from parts_agent.app import app

if __name__ == "__main__":
    app.run()
'''


class PartsAgentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        artifact_kind = self.node.try_get_context("artifact") or "code"
        if artifact_kind == "container":
            artifact = agentcore.AgentRuntimeArtifact.from_asset(
                str(REPO_ROOT),
                platform=ecr_assets.Platform.LINUX_ARM64,
                exclude=[".venv", ".git", ".agent", "infra", "tests", "evals", "**/__pycache__"],
            )
        elif artifact_kind == "code":
            artifact = agentcore.AgentRuntimeArtifact.from_code_asset(
                path=str(REPO_ROOT / "src"),  # only hashes the source; the bundler builds the real zip
                runtime=agentcore.AgentCoreRuntime.PYTHON_3_12,
                entrypoint=["opentelemetry-instrument", "main.py"],
                bundling=BundlingOptions(
                    image=DockerImage.from_registry("unused-local-bundling-only"),
                    local=UvArm64Bundler(),
                ),
                asset_hash=_source_hash(),
            )
        else:
            raise ValueError(f"-c artifact must be 'code' or 'container', got {artifact_kind!r}")

        env = {k: os.environ[k] for k in PASSTHROUGH_ENV if os.environ.get(k)}
        env.update({"PARTS_DATA_DIR": "/var/task/data", "SESSION_DIR": "/tmp/.agent/sessions"})
        if artifact_kind == "container":
            env["PARTS_DATA_DIR"] = "/app/data"

        runtime = agentcore.Runtime(
            self,
            "PartsAgentRuntime",
            runtime_name="parts_agent",
            description="Parts-counter agent + safety monitor (Strands harness)",
            agent_runtime_artifact=artifact,
            environment_variables=env,
            # X-Ray trace delivery needs CloudWatch Transaction Search enabled in the account first
            # (once per account): https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html
            tracing_enabled=self.node.try_get_context("tracing") in ("true", True),
        )

        # Every model call goes to Bedrock; cross-region inference profiles need the profile and the model.
        runtime.add_to_role_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[
                    "arn:aws:bedrock:*::foundation-model/*",
                    f"arn:aws:bedrock:*:{self.account}:inference-profile/*",
                    # OpenAI models on Bedrock (the classifier) also check the account's default project.
                    f"arn:aws:bedrock:*:{self.account}:project/default",
                ],
            )
        )

        CfnOutput(self, "RuntimeArn", value=runtime.agent_runtime_arn)
        CfnOutput(self, "RuntimeId", value=runtime.agent_runtime_id)


@jsii.implements(ILocalBundling)
class UvArm64Bundler:
    """Builds the direct-code-deployment zip on this machine, no Docker.

    Installs Linux ARM64 wheels of every dependency with `uv pip install --python-platform`, then copies
    the package, the catalog and service data, and a main.py entrypoint next to them.
    """

    def try_bundle(self, output_dir: str, *args, **kwargs) -> bool:
        out = Path(output_dir)
        if not shutil.which("uv"):
            raise RuntimeError(
                "Building the AgentCore code zip needs uv: https://docs.astral.sh/uv/getting-started/installation/"
            )
        subprocess.run(
            [
                "uv", "pip", "install",
                # AgentCore runs Amazon Linux 2023 (glibc 2.34) on ARM64. Some strands-harness
                # dependencies only ship manylinux_2_28 wheels, so target that instead of manylinux2014.
                "--python-platform", "aarch64-manylinux_2_28",
                "--python-version", PYTHON_VERSION,
                "--only-binary", ":all:",
                "--target", str(out),
                "--quiet",
                *_runtime_requirements(),
            ],
            check=True,
        )
        shutil.copytree(REPO_ROOT / "src" / "parts_agent", out / "parts_agent", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO_ROOT / "data", out / "data")
        (out / "main.py").write_text(MAIN_PY)
        for path in out.rglob("__pycache__"):
            shutil.rmtree(path, ignore_errors=True)
        return True


def _runtime_requirements() -> list[str]:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    return [*project["dependencies"], *project["optional-dependencies"]["deploy"]]


def _source_hash() -> str:
    """Rebuild the zip only when the code, data or dependencies change."""
    import hashlib

    digest = hashlib.sha256()
    for base in ("src", "data"):
        for path in sorted((REPO_ROOT / base).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(REPO_ROOT)).encode())
                digest.update(path.read_bytes())
    digest.update((REPO_ROOT / "pyproject.toml").read_bytes())
    return digest.hexdigest()
