from dataclasses import dataclass


@dataclass
class DeploymentEnvironmentConfig:
    site_domain_name: str
