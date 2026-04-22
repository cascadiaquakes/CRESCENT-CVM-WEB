## Workflow

The github workflow in `main.yml` assumes a [trunk-based development workflow](https://trunkbaseddevelopment.com/) where developers either push directly to the `main` branch or create PRs from short-lived feature branches that are merged into `main` (typically after code review).

The github workflow includes 4 jobs:

```
Validate Docker Image Build --> Build and Push Docker --> Deploy Dev Resources --> Deploy to Prod Environment
```

Pushing to `main` or merging a PR to `main` will trigger a deploy of the Dev resources automatically.

The 'Deploy to Prod Environment' job is gated by a GitHub Environment `Production`. The GitHub repository settings for the Environment include the list of **Required reviewers** who are able to trigger the prod deployment job.

The CDK App uses a `--context` variable to set the deployment environment.  Dev and Prod resources are created in the same AWS account.  Dev resources will be created with a `dev-` prefix.

### How to Deploy to Prod

The workflow will pause for approval after a successful Dev deployment and before 'Deploy to Prod Environmnet' can run.

After a Reviewer approves the pending deployment, the 'Deploy to Prod Environment' job will run which deploys to production the infrastructure and the version of software that was previously built in this workflow run.



## Docker image tagging and the CDK App

When the workflow pushes a Docker image, that image is tagged with specific version info (commit sha, custom version identifier that includes the timestamp and short sha).  This full reference to the versioned image will be populated into the `ECR_IMAGE_URI` environment variable for later use. The image will also be tagged with `dev-latest`.

When a reviewer approves the pending deployment, 'Deploy to Prod Environment' will run and the image already created in this workflow receives an additional tag `prod-latest` (the image is not rebuilt again). This helps avoid differences between Dev and Prod.

The CDK app will reference the specific version of the image via the environment variable `ECR_IMAGE_URI`.

The `dev-latest` and `prod-latest` tags are a convenience for humans.  They will also be used by the CDK app if `ECR_IMAGE_URI` is unset.



## CDK App Migration

The CDK App contained in this directory was created by the following migration process:

1. Export the AWS CloudFormation Template from the existing deployed application (`aws cloudformation get-template ...`)
2. Strip the template of cross-account resource references, etc. to create a modified CloudFormation template.
3. Run `cdk migrate` to create a new CDK app from the modified CloudFormation template.
4. Make manual changes to the new CDK app to allow deployments to multiple environments (Dev,Prod).
