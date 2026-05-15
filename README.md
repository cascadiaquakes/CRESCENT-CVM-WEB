# CRESCENT CVM-Web

**CVM-Web** is a web-based application for previewing, visualizing, and extracting data from the Community Velocity Model (CVM) compiled by the Cascadia Region Earthquake Science Center ([CRESCENT](https://cascadiaquakes.org/about-crescent/)). The CVM is a curated collection of seismic velocity models covering the Cascadia Subduction Zone and surrounding region. This platform provides an interactive interface — powered by [CesiumJS](https://cesium.com/platform/cesiumjs/) 3D maps and the [FastAPI](https://fastapi.tiangolo.com/) web framework — for exploring model metadata, comparing model coverage, generating slice/cross-section/depth-profile previews, and downloading subsets of model data. The models are compatible with the Python tooling provided by the companion [CVM-Tools](https://cascadiaquakes.github.io/cvm-tools-book/index.html) project, and are also overlaid with current [Community Fault Model (CFM)](https://github.com/cascadiaquakes/crescent-cfm) traces and surfaces.

## Features

- Interactive Cesium 3D map showing CVM model coverage, CFM fault traces and surfaces, the Cascadia subduction interface, regional seismicity, and US/Canada boundaries.
- Browsable repository of 3D and 2D velocity models with full metadata (geospatial extents, resolution, variables, references).
- Filtering of models by geographic area, depth range, and available parameters; up-to-five-model coverage comparison.
- Visualization tools:
  - **Cross-Section Viewer** — vertical slices through a model along an arbitrary path.
  - **Horizontal-Slice Viewer** — constant-depth slices through a model.
  - **Depth-Profile Viewer** — vertical profiles at a single point.
- Data extraction tools that produce downloadable subsets in netCDF, CSV, and GeoCSV formats:
  - **Volume Data** — 3D subvolumes.
  - **Cross-Section Data** — vertical-slice data along a user-defined path.
  - **Depth-Slice Data** — horizontal-slice data at a chosen depth.
- Cloud-backed extraction for large requests via AWS Lambda functions reading from S3-hosted netCDF.
- Configurable Cesium Ion access token retrieved from the server at runtime.

## Directory Structure

```
CRESCENT-CVM-WEB/
├── app/
│   │                               # Application logic (FastAPI app, routers, utilities)
│   ├── static/                     # Static assets such as CSS, JS, images, and GeoJSON boundaries
│   │   ├── boundary_geojson/       # US/Canada administrative boundary GeoJSON
│   │   ├── config/                 # Client-side configuration (project extents, repository config, news)
│   │   ├── css/                    # Stylesheets
│   │   ├── images/                 # CRESCENT/NSF logos and icons
│   │   └── js/                     # Shared JS and 3D helpers
│   │
│   ├── templates/                  # Jinja2 HTML templates rendered by FastAPI
│   │   ├── index.html              # Repository / landing page
│   │   ├── x-section-viewer.html   # Cross-section viewer
│   │   ├── depth-slice-viewer.html # Depth-slice viewer
│   │   ├── depth-profile-viewer.html
│   │   ├── volume-data.html        # Volume extraction form
│   │   └── ...                     # Additional viewer, 3D, and extraction pages
│   │
│   ├── routes/                     # FastAPI route handlers
│   │   ├── main_routes.py          # Landing page, model listings, Cesium token, image generation
│   │   ├── vis_routes.py           # Viewer page routes (/vis/*)
│   │   └── data_routes.py          # Data extraction routes (/data/*)
│   │
│   ├── utils/                      # Helper modules
│   ├── constants.py                # Environment-specific Lambda/S3 names, prefixes, version
│   ├── depthslice.py               # Depth-slice computation helpers
│   └── main.py                     # FastAPI app entry point
├── infra/                          # Infrastructure-as-Code (AWS CDK app, CI/CD notes)
├── docker-compose.yml              # Local development compose file
├── Dockerfile                      # Container build configuration
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Getting Started

### Prerequisites

- Python 3.13
- Docker (optional; recommended for a container-parity workflow)
- AWS credentials with read access to the CVM S3 bucket and invoke access to the extraction Lambda functions (required for S3-backed routes and large-data extraction)
- A [Cesium Ion](https://cesium.com/ion/) access token (served to the client by the `/get-token` endpoint)

### Configure a local `.env` file

Both the Docker and venv workflows read configuration from a `.env` file in the
repository root. Create one with the variables described under
[Environment Variables](#environment-variables); a minimal example:

```bash
ENVIRONMENT=dev
CESIUM_KEYS={"cesium_access_token":"<your-cesium-ion-token>"}

AWS_DEFAULT_REGION=us-east-2
AWS_ACCESS_KEY_ID=<your-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret>
# AWS_SESSION_TOKEN=<token>   # only needed for temporary/SSO credentials
```

`.env` is gitignored — do not commit it.

### Run Locally with Docker

```bash
docker compose down # Only needed if the build is stale
docker compose up --build
```

`docker-compose.yml` loads `.env` into the container automatically. The app will
be available at <http://localhost:8000>.

### Run Locally without Docker

Create and activate a virtual environment, install dependencies, then load
`.env` into your shell before starting uvicorn (uvicorn does not auto-load
`.env`):

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

set -a; source .env; set +a       # export every variable in .env
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The app will be available at <http://localhost:8000>.

### Environment Variables

- `ENVIRONMENT` — selects the deployment environment (`dev` or `crescent`). This controls which S3 bucket and Lambda function names from `app/constants.py` are used. Defaults to `dev`.
- `CESIUM_KEYS` — JSON string of Cesium Ion access tokens served to the client by `/get-token`, of the form `{"cesium_access_token":"<token>"}`. To obtain a token, sign up or log in at <https://cesium.com/ion/> and open the **Access Tokens** tab in the dashboard — use the auto-created default token or create a new one. In deployed environments this is injected from AWS Secrets Manager; for local development, set it directly in `.env`.
- AWS credentials — local development requires `AWS_REGION` plus either `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN` for temporary credentials) in `.env`. In deployed environments these are supplied by the attached IAM role. If credentials are missing, boto3 will fall through to the EC2 instance-metadata service and stall with timeout errors on every S3 or Lambda call.

## Data

CVM models are stored as netCDF files in an S3 bucket, with companion JSON metadata and GeoJSON coverage outlines. The bucket name and key prefixes are defined per-environment in `app/constants.py`:

```
data/netcdf/   # model netCDF files
data/json/     # model metadata
data/geojson/  # model coverage outlines
data/config    # repository-wide configuration
```

To prepare or contribute models for inclusion in the repository, refer to the companion projects:

- 🔗 [CVM-Tools](https://github.com/cascadiaquakes/cvm-tools) — Python tools and conventions for building CVM-compliant netCDF files. See also the [CVM-Tools Book](https://cascadiaquakes.github.io/cvm-tools-book/index.html).
- 🔗 [CRESCENT-CFM](https://github.com/cascadiaquakes/crescent-cfm) — Community Fault Model GeoJSON files overlaid on the CVM-Web map.

## Deployment

The application is deployed to AWS via a CDK app in `infra/cvmwebApp513F4FCD/`. CI/CD is handled by the GitHub Actions workflow in `.github/workflows/main.yml`, which builds and pushes a Docker image to ECR, deploys to a `dev` environment automatically on merges to `main`, and gates production deployment behind a GitHub Environment approval. See `infra/README.md` for the deployment workflow, image tagging conventions, and notes on the original CDK migration.

## Contact

For questions, feedback, or contributions, please visit the [CRESCENT contact page](https://cascadiaquakes.org/about-crescent/).

## Disclaimers

**Data Disclaimer**:
Unless otherwise stated, all data, metadata, and related materials are considered to satisfy the quality standards relative to the purpose for which the data were collected. Although these data and associated metadata have been reviewed for accuracy and completeness, no warranty expressed or implied is made regarding the display or utility of the data for other purposes, nor on all computer systems, nor shall the act of distribution constitute any such warranty.

**Nonendorsement Disclaimer**:
Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the providers of this application or their sponsors.
