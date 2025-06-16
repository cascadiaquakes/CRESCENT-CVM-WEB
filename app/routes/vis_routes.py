import os

from fastapi import APIRouter, Request, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


"""Routes to load visualization pages"""

router = APIRouter()

# Get the absolute path of the directory where the script is located
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Get the parent directory
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# Set the base directory to the parent directory
BASE_DIR = PARENT_DIR
JSON_DIR = os.path.join(BASE_DIR, "static", "json")
NETCDF_DIR = os.path.join(BASE_DIR, "static", "netcdf")


# Templates
templates = Jinja2Templates(directory="templates")


@router.get("/volume-3d", response_class=HTMLResponse)
async def volume_3d(request: Request):
    return templates.TemplateResponse("volume-3d.html", {"request": request})


# Index page.
@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Cross-section viewer.
@router.get("/x-section-viewer", response_class=HTMLResponse)
async def Cross_section_viewer(request: Request):
    return templates.TemplateResponse("x-section-viewer.html", {"request": request})


# Cross-section 3D map.
@router.get("/x-section-3d", response_class=HTMLResponse)
async def Cross_section_3d(request: Request):
    return templates.TemplateResponse("x-section-3d.html", {"request": request})


# Depth-profile viewer.
@router.get("/depth-profile-viewer", response_class=HTMLResponse)
async def depth_profile_viewer(request: Request):
    return templates.TemplateResponse("depth-profile-viewer.html", {"request": request})


# Depth-profile 3D map.
@router.get("/depth-profile-3d", response_class=HTMLResponse)
async def depth_profil_3d(request: Request):
    return templates.TemplateResponse("depth-profile-3d.html", {"request": request})


# Horizontal-slice viewer.
@router.get("/horizontal-slice-viewer", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse(
        "horizontal-slice-viewer.html", {"request": request}
    )


# Depth-slice 3D map.
@router.get("/depth-slice-3d", response_class=HTMLResponse)
async def depth_slice_3d(request: Request):
    return templates.TemplateResponse("depth-slice-3d.html", {"request": request})
