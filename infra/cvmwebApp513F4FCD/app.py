#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cvmweb_app513_f4_fcd.cvmweb_app513_f4_fcd_stack import CvmwebApp513F4FcdStack
from lib.deployment_environment_config import DeploymentEnvironmentConfig

AWS_ACCOUNT_ID = '818214664804'
AWS_REGION = 'us-east-2'
ECR_REPOSITORY_NAME = os.getenv("ECR_REPOSITORY_NAME", default='cvm-web')
ECR_REGISTRY_DOMAIN_NAME = os.getenv("ECR_REGISTRY_DOMAIN_NAME",
                                     default=f'{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com')

# Define configuration based on the deployment environment.
# NOTE: The `deployment-environment` MUST be provided at runtime via CDK context.
# Example: `cdk synth --context deployment-environment=dev`

deployment_environments_configs: dict[str, DeploymentEnvironmentConfig] = {
    "dev": DeploymentEnvironmentConfig(
        site_domain_name = "dev.cvm.cascadiaquakes.org",
        ecr_image_uri = os.getenv("ECR_IMAGE_URI",
                           default=f'{ECR_REGISTRY_DOMAIN_NAME}/{ECR_REPOSITORY_NAME}:dev-latest'),
        bucket_name = "cvm-s3-data-dev-us-east-2-aer1lu3eichu",
        lambda_function_name_slice = "cvm-data-extractor-latest-extract-slice",
        lambda_function_name_xsection = "cvm-data-extractor-latest-extract-xsection",
        lambda_function_name_volume = "cvm-data-extractor-latest-extract-volume",
        runtime_environment_name = "dev"
    ),
    "prod": DeploymentEnvironmentConfig(
        site_domain_name = "cvm.cascadiaquakes.org",
        ecr_image_uri = os.getenv("ECR_IMAGE_URI",
                           default=f'{ECR_REGISTRY_DOMAIN_NAME}/{ECR_REPOSITORY_NAME}:prod-latest'),
        bucket_name = "cvm-s3-data-crescent-us-east-2-aer1lu3eichu",
        lambda_function_name_slice = "cvm-data-extractor-extract-slice",
        lambda_function_name_xsection = "cvm-data-extractor-extract-xsection",
        lambda_function_name_volume = "cvm-data-extractor-extract-volume",
        runtime_environment_name = "crescent"
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
    env=cdk.Environment(account=AWS_ACCOUNT_ID, region=AWS_REGION),
    config=env_config
    )

app.synth()
