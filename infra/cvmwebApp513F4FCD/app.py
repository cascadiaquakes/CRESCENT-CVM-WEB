#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cvmweb_app513_f4_fcd.cvmweb_app513_f4_fcd_stack import CvmwebApp513F4FcdStack
from lib.deployment_environment_config import DeploymentEnvironmentConfig

# Define configuration based on the deployment environment.
# NOTE: The `deployment-environment` MUST be provided at runtime via CDK context.
# Example: `cdk synth --context deployment-environment=dev`

deployment_environments_configs: dict[str, DeploymentEnvironmentConfig] = {
    "dev": DeploymentEnvironmentConfig(
        site_domain_name = "dev.cvm.cascadiaquakes.org",
    ),
    "prod": DeploymentEnvironmentConfig(
        site_domain_name = "cvm.cascadiaquakes.org",
    )
}


app = cdk.App()
# Get runtime deployment environment from CDK context
deployment_environment = app.node.try_get_context("deployment-environment")

if not deployment_environment:
    raise SystemExit(
        "Error: deployment-environment context variable not set."
        " Use '--context deployment-environment=<name of deployment environment>' to set it."
    )

env_config = deployment_environments_configs.get(deployment_environment)

if not env_config:
    raise SystemExit(
        f"Error: No configuration found for deployment environment '{deployment_environment}'."
    )

# CDK uses the stack name as the prefix for nearly all resources created.
# Create a prefix for non-Prod Stacks to avoid naming conflicts between environments.
resource_prefix=f"{deployment_environment}-" if deployment_environment != "prod" else ""

CvmwebApp513F4FcdStack(
    app,
    f"{resource_prefix}cvmwebApp513F4FCD",
    env=cdk.Environment(account='818214664804', region='us-east-2'),
    config=env_config
    )

app.synth()
