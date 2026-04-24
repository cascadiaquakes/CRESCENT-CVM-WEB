from aws_cdk import Stack
import aws_cdk as cdk
import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_ecs as ecs
import aws_cdk.aws_elasticloadbalancingv2 as elasticloadbalancingv2
import aws_cdk.aws_iam as iam
import aws_cdk.aws_logs as logs
import aws_cdk.aws_route53 as route53
from constructs import Construct
from lib.deployment_environment_config import DeploymentEnvironmentConfig

"""

"""
class CvmwebApp513F4FcdStack(Stack):
  def __init__(self, scope: Construct, construct_id: str, config: DeploymentEnvironmentConfig, **kwargs) -> None:
    super().__init__(scope, construct_id, **kwargs)

    # Resources
    albFargateServiceLbPublicListenerEcsGroupE1c21d93 = elasticloadbalancingv2.CfnTargetGroup(self, 'AlbFargateServiceLBPublicListenerECSGroupE1C21D93',
          health_check_interval_seconds = 5,
          health_check_path = '/',
          health_check_timeout_seconds = 3,
          healthy_threshold_count = 2,
          port = 80,
          protocol = 'HTTP',
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
          target_group_attributes = [
            {
              'key': 'stickiness.enabled',
              'value': 'false',
            },
            {
              'key': 'deregistration_delay.timeout_seconds',
              'value': '5',
            },
          ],
          target_type = 'ip',
          vpc_id = 'vpc-015db383b6e754567',
        )

    albFargateServiceLbSecurityGroup051D028d = ec2.CfnSecurityGroup(self, 'AlbFargateServiceLBSecurityGroup051D028D',
          group_description = 'Automatically created Security Group for ELB cvmwebAppAlbFargateServiceLB91E9B416',
          security_group_ingress = [
            {
              'cidrIp': '0.0.0.0/0',
              'description': 'Allow from anyone on port 443',
              'fromPort': 443,
              'ipProtocol': 'tcp',
              'toPort': 443,
            },
            {
              'cidrIp': '0.0.0.0/0',
              'description': 'Allow from anyone on port 80',
              'fromPort': 80,
              'ipProtocol': 'tcp',
              'toPort': 80,
            },
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
          vpc_id = 'vpc-015db383b6e754567',
        )

    fargateCluster76631Ea2 = ecs.CfnCluster(self, 'FargateCluster76631EA2',
          cluster_settings = [
            {
              'name': 'containerInsights',
              'value': 'disabled',
            },
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
        )

    fargateTaskAppLogGroupFdfbef61 = logs.CfnLogGroup(self, 'FargateTaskAppLogGroupFDFBEF61',
          retention_in_days = 90,
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
        )
    fargateTaskAppLogGroupFdfbef61.cfn_options.deletion_policy = cdk.CfnDeletionPolicy.RETAIN

    fargateTaskExecutionRole2B907e8a = iam.CfnRole(self, 'FargateTaskExecutionRole2B907E8A',
          assume_role_policy_document = {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'ecs-tasks.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
        )

    fargateTaskTaskRoleB7c51ec5 = iam.CfnRole(self, 'FargateTaskTaskRoleB7C51EC5',
          assume_role_policy_document = {
            'Statement': [
              {
                'Action': 'sts:AssumeRole',
                'Effect': 'Allow',
                'Principal': {
                  'Service': 'ecs-tasks.amazonaws.com',
                },
              },
            ],
            'Version': '2012-10-17',
          },
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
        )

    albFargateServiceLbd293316e = elasticloadbalancingv2.CfnLoadBalancer(self, 'AlbFargateServiceLBD293316E',
          load_balancer_attributes = [
            {
              'key': 'deletion_protection.enabled',
              'value': 'false',
            },
          ],
          scheme = 'internet-facing',
          security_groups = [
            albFargateServiceLbSecurityGroup051D028d.attr_group_id,
          ],
          subnets = [
            'subnet-017479231955d2800',
            'subnet-007fd3a7ac6cabe4c',
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
          type = 'application',
        )

    fargateClusterA52aec7a = ecs.CfnClusterCapacityProviderAssociations(self, 'FargateClusterA52AEC7A',
          capacity_providers = [
            'FARGATE',
            'FARGATE_SPOT',
          ],
          cluster = fargateCluster76631Ea2.ref,
          default_capacity_provider_strategy = [
          ],
        )
    fargateClusterA52aec7a.add_dependency(fargateCluster76631Ea2)

    fargateTaskB499db9f = ecs.CfnTaskDefinition(self, 'FargateTaskB499DB9F',
          container_definitions = [
            {
              'environment': [
                {
                  'name': 'ENVIRONMENT',
                  'value': config.runtime_environment_name,
                },
              ],
              'essential': True,
              'image': config.ecr_image_uri,
              'logConfiguration': {
                'logDriver': 'awslogs',
                'options': {
                  'awslogs-group': fargateTaskAppLogGroupFdfbef61.ref,
                  'awslogs-stream-prefix': 'cvm-web-crescent-cvm-web-web',
                  'awslogs-region': 'us-east-2',
                },
              },
              'name': 'cvm-web-web',
              'portMappings': [
                {
                  'containerPort': 80,
                  'protocol': 'tcp',
                },
              ],
              'secrets': [
                {
                  'name': 'CESIUM_KEYS',
                  'valueFrom': 'arn:aws:secretsmanager:us-east-2:818214664804:secret:crescent-cesium-secrets-TPAkBS',
                },
              ],
            },
          ],
          cpu = '1024',
          execution_role_arn = fargateTaskExecutionRole2B907e8a.attr_arn,
          family = 'cvmwebAppFargateTask71CE40FA',
          memory = '8192',
          network_mode = 'awsvpc',
          requires_compatibilities = [
            'FARGATE',
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
          task_role_arn = fargateTaskTaskRoleB7c51ec5.attr_arn,
        )

    fargateTaskExecutionRoleDefaultPolicy695Cbb26 = iam.CfnPolicy(self, 'FargateTaskExecutionRoleDefaultPolicy695CBB26',
          policy_document = {
            'Statement': [
              {
                'Action': [
                  'ecr:BatchCheckLayerAvailability',
                  'ecr:BatchGetImage',
                  'ecr:GetDownloadUrlForLayer',
                ],
                'Effect': 'Allow',
                'Resource': [
                  'arn:aws:ecr:us-east-2:818214664804:repository/cvm-web',
                ],
              },
              {
                'Action': 'ecr:GetAuthorizationToken',
                'Effect': 'Allow',
                'Resource': '*',
              },
              {
                'Action': [
                  'logs:CreateLogStream',
                  'logs:PutLogEvents',
                ],
                'Effect': 'Allow',
                'Resource': [
                  fargateTaskAppLogGroupFdfbef61.attr_arn,
                ],
              },
              {
                'Action': [
                  'secretsmanager:DescribeSecret',
                  'secretsmanager:GetSecretValue',
                ],
                'Effect': 'Allow',
                'Resource': 'arn:aws:secretsmanager:us-east-2:818214664804:secret:crescent-cesium-secrets-TPAkBS',
              },
            ],
            'Version': '2012-10-17',
          },
          policy_name = 'FargateTaskExecutionRoleDefaultPolicy695CBB26',
          roles = [
            fargateTaskExecutionRole2B907e8a.ref,
          ],
        )

    fargateTaskTaskRoleDefaultPolicy343D3d72 = iam.CfnPolicy(self, 'FargateTaskTaskRoleDefaultPolicy343D3D72',
          policy_document = {
            'Statement': [
              {
                'Action': [
                  'aps:RemoteWrite',
                  'logs:CreateLogStream',
                  'logs:DescribeLogGroups',
                  'logs:DescribeLogStreams',
                  'logs:PutLogEvents',
                  'ssmmessages:CreateControlChannel',
                  'ssmmessages:CreateDataChannel',
                  'ssmmessages:OpenControlChannel',
                  'ssmmessages:OpenDataChannel',
                  'xray:GetSamplingRules',
                  'xray:GetSamplingStatisticSummaries',
                  'xray:GetSamplingTargets',
                  'xray:PutTelemetryRecords',
                  'xray:PutTraceSegments',
                ],
                'Effect': 'Allow',
                'Resource': '*',
              },
              {
                'Action': 'lambda:InvokeFunction',
                'Effect': 'Allow',
                'Resource': [
                  f'arn:aws:lambda:us-east-2:818214664804:function:{config.lambda_function_name_slice}',
                  f'arn:aws:lambda:us-east-2:818214664804:function:{config.lambda_function_name_xsection}',
                  f'arn:aws:lambda:us-east-2:818214664804:function:{config.lambda_function_name_volume}',
                ],
              },
              {
                'Action': [
                  's3:DeleteObject',
                  's3:GetObject',
                  's3:ListBucket',
                  's3:PutObject',
                ],
                'Effect': 'Allow',
                'Resource': [
                  f'arn:aws:s3:::{config.bucket_name}',
                  f'arn:aws:s3:::{config.bucket_name}/data/*',
                  f'arn:aws:s3:::{config.bucket_name}/work_area/*',
                ],
              },
            ],
            'Version': '2012-10-17',
          },
          policy_name = 'FargateTaskTaskRoleDefaultPolicy343D3D72',
          roles = [
            fargateTaskTaskRoleB7c51ec5.ref,
          ],
        )

    albFargateServiceDnsd893ec68 = route53.CfnRecordSet(self, 'AlbFargateServiceDNSD893EC68',
          alias_target = {
            'dnsName': ''.join([
              'dualstack.',
              albFargateServiceLbd293316e.attr_dns_name,
            ]),
            'hostedZoneId': albFargateServiceLbd293316e.attr_canonical_hosted_zone_id,
          },
          hosted_zone_id = 'Z09086153UR2GVATV2P2R',
          name = f"{config.site_domain_name}.",
          type = 'A',
        )

    albFargateServiceLbPublicListenerCb0a3f85 = elasticloadbalancingv2.CfnListener(self, 'AlbFargateServiceLBPublicListenerCB0A3F85',
          certificates = [
            {
              'certificateArn': 'arn:aws:acm:us-east-2:818214664804:certificate/69cc847f-bee7-4bdb-9a5d-3f5f7731c9ac',
            },
          ],
          default_actions = [
            {
              'targetGroupArn': albFargateServiceLbPublicListenerEcsGroupE1c21d93.ref,
              'type': 'forward',
            },
          ],
          load_balancer_arn = albFargateServiceLbd293316e.ref,
          port = 443,
          protocol = 'HTTPS',
        )

    albFargateServiceLbPublicRedirectListenerC5f4d8dc = elasticloadbalancingv2.CfnListener(self, 'AlbFargateServiceLBPublicRedirectListenerC5F4D8DC',
          default_actions = [
            {
              'redirectConfig': {
                'port': '443',
                'protocol': 'HTTPS',
                'statusCode': 'HTTP_301',
              },
              'type': 'redirect',
            },
          ],
          load_balancer_arn = albFargateServiceLbd293316e.ref,
          port = 80,
          protocol = 'HTTP',
        )

    albFargateServiceSecurityGroup81F44dcf = ec2.CfnSecurityGroup(self, 'AlbFargateServiceSecurityGroup81F44DCF',
          group_description = 'cvm-web/App/AlbFargateService/Service/SecurityGroup',
          security_group_egress = [
            {
              'cidrIp': '0.0.0.0/0',
              'description': 'Allow all outbound traffic by default',
              'ipProtocol': '-1',
            },
          ],
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
          vpc_id = 'vpc-015db383b6e754567',
        )
    albFargateServiceSecurityGroup81F44dcf.add_dependency(fargateClusterA52aec7a)
    albFargateServiceSecurityGroup81F44dcf.add_dependency(fargateTaskTaskRoleDefaultPolicy343D3d72)
    albFargateServiceSecurityGroup81F44dcf.add_dependency(fargateTaskTaskRoleB7c51ec5)

    albFargateServiceC7b07899 = ecs.CfnService(self, 'AlbFargateServiceC7B07899',
          capacity_provider_strategy = [
            {
              'capacityProvider': 'FARGATE_SPOT',
              'weight': 0,
            },
            {
              'base': 1,
              'capacityProvider': 'FARGATE',
              'weight': 1,
            },
          ],
          cluster = fargateCluster76631Ea2.ref,
          deployment_configuration = {
            'alarms': {
              'alarmNames': [
              ],
              'enable': False,
              'rollback': False,
            },
            'maximumPercent': 200,
            'minimumHealthyPercent': 75,
          },
          desired_count = 1,
          enable_ecs_managed_tags = False,
          enable_execute_command = True,
          health_check_grace_period_seconds = 60,
          load_balancers = [
            {
              'containerName': 'cvm-web-web',
              'containerPort': 80,
              'targetGroupArn': albFargateServiceLbPublicListenerEcsGroupE1c21d93.ref,
            },
          ],
          network_configuration = {
            'awsvpcConfiguration': {
              'assignPublicIp': 'DISABLED',
              'securityGroups': [
                albFargateServiceSecurityGroup81F44dcf.attr_group_id,
              ],
              'subnets': [
                'subnet-071357450c2997439',
                'subnet-0868a41e2f54b512f',
              ],
            },
          },
          propagate_tags = 'SERVICE',
          tags = [
            {
              'key': 'crescent:application:name',
              'value': 'CVM-WEB',
            },
          ],
          task_definition = fargateTaskB499db9f.ref,
        )
    albFargateServiceC7b07899.add_dependency(albFargateServiceLbPublicListenerEcsGroupE1c21d93)
    albFargateServiceC7b07899.add_dependency(albFargateServiceLbPublicListenerCb0a3f85)
    albFargateServiceC7b07899.add_dependency(fargateClusterA52aec7a)
    albFargateServiceC7b07899.add_dependency(fargateTaskTaskRoleDefaultPolicy343D3d72)
    albFargateServiceC7b07899.add_dependency(fargateTaskTaskRoleB7c51ec5)

    albFargateServiceLbSecurityGrouptocvmwebAppAlbFargateServiceSecurityGroup954D937a80bd35822f = ec2.CfnSecurityGroupEgress(self, 'AlbFargateServiceLBSecurityGrouptocvmwebAppAlbFargateServiceSecurityGroup954D937A80BD35822F',
          description = 'Load balancer to target',
          destination_security_group_id = albFargateServiceSecurityGroup81F44dcf.attr_group_id,
          from_port = 80,
          group_id = albFargateServiceLbSecurityGroup051D028d.attr_group_id,
          ip_protocol = 'tcp',
          to_port = 80,
        )

    albFargateServiceSecurityGroupfromcvmwebAppAlbFargateServiceLbSecurityGroup12184Fd2803ceb0082 = ec2.CfnSecurityGroupIngress(self, 'AlbFargateServiceSecurityGroupfromcvmwebAppAlbFargateServiceLBSecurityGroup12184FD2803CEB0082',
          description = 'Load balancer to target',
          from_port = 80,
          group_id = albFargateServiceSecurityGroup81F44dcf.attr_group_id,
          ip_protocol = 'tcp',
          source_security_group_id = albFargateServiceLbSecurityGroup051D028d.attr_group_id,
          to_port = 80,
        )
    albFargateServiceSecurityGroupfromcvmwebAppAlbFargateServiceLbSecurityGroup12184Fd2803ceb0082.add_dependency(fargateClusterA52aec7a)
    albFargateServiceSecurityGroupfromcvmwebAppAlbFargateServiceLbSecurityGroup12184Fd2803ceb0082.add_dependency(fargateTaskTaskRoleDefaultPolicy343D3d72)
    albFargateServiceSecurityGroupfromcvmwebAppAlbFargateServiceLbSecurityGroup12184Fd2803ceb0082.add_dependency(fargateTaskTaskRoleB7c51ec5)

    # Outputs
    self.alb_fargate_service_load_balancer_dns76f2ca58 = albFargateServiceLbd293316e.attr_dns_name
    cdk.CfnOutput(self, 'CfnOutputAlbFargateServiceLoadBalancerDNS76F2CA58', 
      key = 'AlbFargateServiceLoadBalancerDNS76F2CA58',
      value = str(self.alb_fargate_service_load_balancer_dns76f2ca58),
    )

    self.alb_fargate_service_service_urlc151c919 = ''.join([
      'https://',
      albFargateServiceDnsd893ec68.ref,
    ])
    cdk.CfnOutput(self, 'CfnOutputAlbFargateServiceServiceURLC151C919', 
      key = 'AlbFargateServiceServiceURLC151C919',
      value = str(self.alb_fargate_service_service_urlc151c919),
    )



