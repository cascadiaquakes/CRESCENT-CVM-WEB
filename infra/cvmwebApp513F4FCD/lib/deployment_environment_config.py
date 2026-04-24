from dataclasses import dataclass

@dataclass
class DeploymentEnvironmentConfig:
    site_domain_name: str
    ecr_image_uri: str
    bucket_name: str
    lambda_function_name_slice: str
    lambda_function_name_xsection: str
    lambda_function_name_volume: str
    runtime_environment_name: str