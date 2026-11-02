#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stack import PartsAgentStack

app = cdk.App()
PartsAgentStack(
    app,
    "PartsAgentStack",
    env=cdk.Environment(account=os.environ.get("CDK_DEFAULT_ACCOUNT"), region=os.environ.get("CDK_DEFAULT_REGION")),
)
app.synth()
