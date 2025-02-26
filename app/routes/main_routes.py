import os
import sys
import json
import traceback
import boto3

from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from utils.json_utilities import read_json_files
import xarray as xr
from pyproj import Proj, Geod
from numpy import nanmin, nanmax

# from metpy.interpolate import cross_section
import base64
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
from datetime import datetime, timezone
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

import math
import gc

import urllib.parse
from urllib.parse import urlencode
from uuid import uuid4

import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from geographiclib.geodesic import Geodesic


router = APIRouter()

# Get the absolute path of the directory where the script is located
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Get the parent directory
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

# Set the base directory to the parent directory
BASE_DIR = PARENT_DIR
JSON_DIR = os.path.join(BASE_DIR, "static", "json")
NETCDF_DIR = os.path.join(BASE_DIR, "static", "netcdf")
LOGO_FILE = "static/images/Crescent_Logos_horizontal_transparent.png"

# Logo placement in inches below the lower left corner of the plot area
LOGO_INCH_OFFSET_BELOW = 0.5

grid_ref_dict = {
    "latitude_longitude": {
        "x": "longitude",
        "y": "latitude",
        "x2": "longitude",
        "y2": "latitude",
    },
    "transverse_mercator": {"x": "x", "y": "y", "x2": "easting", "y2": "northing"},
}

# Templates
templates = Jinja2Templates(directory="templates")


def points_to_array(points, delimiter=";"):
    """Converts a string of coordinate points that are separated by a delimiter to array of points.'123,45;124,46' -> [[123,45],[124,46]]"""
    points_list = points.strip().split(delimiter)
    coord_list = list()
    name_list = list()
    name_info_list = list()
    for point in points_list:
        lon_lat, name, name_info = point.strip().split("|")
        lon, lat = lon_lat.strip().split(",")
        coord_list.append([float(lat), float(lon)])
        name_list.append(name)
        name_info_list.append(name_info)
    return coord_list, name_list, name_info_list


def points_to_dist(start, points_list):
    """calculates distance of a series of points from a starting point."""
    distance_list = list()
    geod = Geod(ellps="WGS84")

    for point in points_list:
        _, _, distance = geod.inv(start[1], start[0], point[1], point[0])
        distance_list.append(distance)
    return distance_list


def utc_now():
    """Return the current UTC time."""
    try:
        _utc = datetime.now(tz=timezone.utc)
        utc = {
            "date_time": _utc.strftime("%Y-%m-%dT%H:%M:%S"),
            "datetime": _utc,
            "epoch": _utc.timestamp(),
        }
        return utc
    except:
        logger.error(f"[ERR] Failed to get the current UTC time")
        raise


def standard_units(unit):
    """Check an input unit and return the corresponding standard unit."""
    unit = unit.strip().lower()
    if unit in ["m", "meter"]:
        return "m"
    elif unit in ["degrees", "degrees_east", "degrees_north"]:
        return "degrees"
    elif unit in ["km", "kilometer"]:
        return "km"
    elif unit in ["g/cc", "g/cm3", "g.cm-3", "grams.centimeter-3"]:
        return "g/cc"
    elif unit in ["kg/m3", "kh.m-3"]:
        return "kg/m3"
    elif unit in ["km/s", "kilometer/second", "km.s-1", "kilometer/s", "km/s"]:
        return "km/s"
    elif unit in ["m/s", "meter/second", "m.s-1", "meter/s", "m/s"]:
        return "m/s"
    elif unit.strip().lower in ["", "none"]:
        return ""


def unit_conversion_factor(unit_in, unit_out):
    """Check input and output unit and return the conversion factor."""

    unit = standard_units(unit_in.strip().lower())
    logger.debug(f"convert units {unit} to {unit_out}")
    if unit in ["m"]:
        if unit_out == "cgs":
            return standard_units("m"), 1
        else:
            return standard_units("km"), 0.001
    elif unit in ["km"]:
        if unit_out == "cgs":
            return standard_units("m"), 1000
        else:
            return standard_units("km"), 1
    elif unit in ["m/s"]:
        if unit_out == "cgs":
            return standard_units("m/s"), 1
        else:
            return standard_units("km/s"), 0.001
    elif unit in ["g/cc"]:
        if unit_out == "cgs":
            return standard_units("g/cc"), 1
        else:
            return standard_units("kg/m3"), 1000
    elif unit in ["km/s"]:
        if unit_out == "cgs":
            return standard_units("m/s"), 1000
        else:
            return standard_units("km/s"), 1
    elif unit in ["kg/m3"]:
        if unit_out == "cgs":
            return standard_units("g/cc"), 0.001
        else:
            return standard_units("kg/m3"), 1
    elif unit in ["", " "]:
        return standard_units(""), 1
    elif unit in ["%"]:
        return standard_units("%"), 1
    elif unit in ["degrees"]:
        return standard_units("degrees"), 1

    else:
        logger.error(f"[ERR] Failed to convert units {unit_in} to {unit_out}")
        return unit_in, 1


def dip_dir_to_azimuth(dip_dir):
    """Convert dip direction to azimuth angle."""
    cardinal_to_azimuth = {
        "N": 0,
        "NE": 45,
        "E": 90,
        "SE": 135,
        "S": 180,
        "SW": 225,
        "W": 270,
        "NW": 315,
    }
    if dip_dir.upper() == "VERTICAL":
        return None  # Special case for vertical dip direction
    return cardinal_to_azimuth[dip_dir.upper()]


def calculate_azimuth(start_lat, start_lon, end_lat, end_lon):
    """Calculate azimuth between two geographical points."""
    geod = Geodesic.WGS84
    inv = geod.Inverse(start_lat, start_lon, end_lat, end_lon)
    return inv["azi1"]


def project_feature(
    dip, dip_azimuth, section_azimuth, upper_depth, lower_depth, start_lat, start_lon
):
    """Project the feature onto the section plane, accounting for dip."""
    dip_angle = np.radians(dip)
    vertical_range = lower_depth - upper_depth

    if dip_azimuth is None:
        # Vertical dip: horizontal positions are the same
        horizontal_position_start = 0
        horizontal_position_end = 0
    else:
        relative_azimuth = np.radians(dip_azimuth - section_azimuth)
        horizontal_offset = vertical_range / np.tan(dip_angle)
        horizontal_position_start = 0  # Start position relative to start coordinates
        horizontal_position_end = horizontal_offset * np.cos(relative_azimuth)

    return horizontal_position_start, horizontal_position_end, upper_depth, lower_depth


def extract_values(dataset, lat, lon, depth_range, interpolation_method='linear', grid_ref="latitude_longitude",
    utm_zone=None,
    ellipsoid=None,):
    """
    Extract values from an xarray dataset for a given latitude, longitude, and depth range 
    using the specified interpolation method.

    Parameters:
    - dataset: xarray Dataset containing latitude, longitude, depth, and the values.
    - lat: The latitude at which to extract the values.
    - lon: The longitude at which to extract the values.
    - depth_range: Tuple (min_depth, max_depth) specifying the depth range.
    - interpolation_method: Method of interpolation (e.g., 'linear', 'nearest', 'cubic').
                            Default is 'linear'.
    
    Returns:
    - interpolated_values: Extracted and interpolated values at the specified lat/lon and within the depth range.
    """

    # Slice the dataset to the selected depth range
    sliced_dataset = dataset.sel(depth=slice(depth_range[0], depth_range[1]))

    # Interpolate values at the given lat and lon with the chosen interpolation method
    if grid_ref == "latitude_longitude":
        interpolated_values = sliced_dataset.interp(
            latitude=lat,
            longitude=lon,
            method=interpolation_method
        )
        logger.debug(f"\n\n\n{grid_ref}>>>>>>>>>{interpolated_values}")
    else:
        if None in (utm_zone, ellipsoid):
            message = f"[ERR] for grid_ref: {grid_ref}, utm_zone and ellipsoid are required. Current values: {utm_zone}, {ellipsoid}!"
            logger.error(message)
            raise

        x, y = project_lonlat_utm(
            lon, lat, utm_zone, ellipsoid=ellipsoid
        )

        # Define a path dataset for interpolation
        interpolated_values = dataset.interp(x=x, y=y, method=interpolation_method)
        logger.debug(f"\n\n\n{grid_ref}>>>>>>>>>{interpolated_values}")

    return interpolated_values



def interpolate_path(
    ds,
    start,
    end,
    num_points=100,
    method="linear",
    grid_ref="latitude_longitude",
    utm_zone=None,
    ellipsoid=None,
):
    """
    Interpolates a dataset along a path defined by start and end coordinates on an irregular grid.

    Parameters:
        ds (xarray.Dataset): The input dataset containing 'latitude' and 'longitude' as coordinates.
        start (tuple): A tuple (latitude, longitude) of the starting point.
        end (tuple): A tuple (latitude, longitude) of the ending point.
        num_points (int): Number of points to interpolate along the path.
        method (str): Interpolation method to use ('linear', 'nearest').

    Returns:
        xarray.Dataset: The interpolated dataset along the path.
    """
    # Create linearly spaced points between start and end
    lat_points = np.linspace(start[0], end[0], num_points)
    lon_points = np.linspace(start[1], end[1], num_points)

    # Define a path dataset for interpolation
    path = xr.Dataset(
        {"latitude": ("points", lat_points), "longitude": ("points", lon_points)}
    )

    # Interpolate the dataset to these points using the specified method
    if grid_ref == "latitude_longitude":
        interpolated_ds = ds.interp(
            latitude=path.latitude, longitude=path.longitude, method=method
        )
    else:
        if None in (utm_zone, ellipsoid):
            message = f"[ERR] for grid_ref: {grid_ref}, utm_zone and ellipsoid are required. Current values: {utm_zone}, {ellipsoid}!"
            logger.error(message)
            raise
        x_points = list()
        y_points = list()
        for index, lat_value in enumerate(lat_points):
            x, y = project_lonlat_utm(
                lon_points[index], lat_points[index], utm_zone, ellipsoid=ellipsoid
            )
            x_points.append(x)
            y_points.append(y)

        # Define a path dataset for interpolation
        path = xr.Dataset({"x": ("points", x_points), "y": ("points", y_points)})
        logger.debug(f"x_points: {x_points}")
        logger.debug(f"y_points: {y_points}")
        interpolated_ds = ds.interp(x=path.x, y=path.y, method=method)

    return interpolated_ds, lat_points, lon_points


def project_lonlat_utm(
    longitude, latitude, utm_zone, ellipsoid, xy_to_latlon=False, preserve_units=False
):
    """
    Performs cartographic transformations. Converts from longitude, latitude to UTM x,y coordinates
    and vice versa using PROJ (https://proj.org).

     Keyword arguments:
    longitude (scalar or array) – Input longitude coordinate(s).
    latitude (scalar or array) – Input latitude coordinate(s).
    xy_to_latlon (bool, default=False) – If inverse is True the inverse transformation from x/y to lon/lat is performed.
    preserve_units (bool) – If false, will ensure +units=m.
    """
    P = Proj(
        proj="utm",
        zone=utm_zone,
        ellps=ellipsoid,
    )
    # preserve_units=preserve_units,

    x, y = P(
        longitude,
        latitude,
        inverse=xy_to_latlon,
    )
    return x, y


def closest(lst, value):
    """Find the closest number in a list to the given value

    Keyword arguments:
    lst -- [required] list of the numbers
    value -- [required] value to find the closest list member for.
    """
    arr = np.array(lst)
    # Check if the array is multi-dimensional
    if arr.ndim > 1:
        # Flatten the array and return
        flat_list = list(arr.flatten())
    else:
        # Return the original array if it's already one-dimensional
        flat_list = lst

    return flat_list[
        min(range(len(flat_list)), key=lambda i: abs(flat_list[i] - value))
    ]


def create_error_image(message: str) -> BytesIO:
    """Generate an image with an error message."""
    # Create an image with white background
    img = Image.new("RGB", (600, 100), color=(255, 255, 255))

    # Initialize the drawing context
    d = ImageDraw.Draw(img)

    # Optionally, add a font (this uses the default PIL font)
    # For custom fonts, use ImageFont.truetype()
    # font = ImageFont.truetype("arial.ttf", 15)

    # Add text to the image
    d.text(
        (10, 10), message, fill=(255, 0, 0)
    )  # Change coordinates and color as needed

    # Save the image to a bytes buffer
    img_io = BytesIO()
    img.save(img_io, "PNG")
    img_io.seek(0)
    image_data = (
        img_io.read()
    )  # Read the entire stream content, which is the image data

    return image_data


def custom_formatter(x):
    """
    Custom formatter function
    If the value is close enough to zero (including -0.0), format it as '0'.
    Otherwise, use the default formatting.
    """
    if abs(x) < 1e-12:  # 1e-12 is used as a threshold for floating-point comparison
        return "0"
    else:
        return f"{x}"


# Route to serve favicon
@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")


@router.get("/request", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@router.get("/news", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("news.html", {"request": request})


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


"""
@router.get("/depth-slice-viewer", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("depth-slice-viewer.html", {"request": request})
"""


# Route to create html for model dropdown. This is a comprehensive list that includes other model parameters.
@router.get("/models_drop_down_coverage", response_class=HTMLResponse)
def models_drop_down_coverage(required_variable: Optional[str] = None):
    logger.debug(f"Received required_variable: {required_variable}")

    json_directory = JSON_DIR
    coords_list = list()
    model_list = list()
    title_list = list()
    logger.debug(f"Reading json_directory {json_directory}")
    for filename in sorted(
        [f for f in os.listdir(json_directory) if f.endswith(".json")]
    ):
        logger.debug(f"Reading {filename}")
        # Opening the file and loading the data
        with open(os.path.join(json_directory, filename), "r") as file:
            json_data = json.load(file)
            logger.debug(f"{json_data}")

            # Skip the file if file does not have the require variable
            if required_variable is not None:
                if required_variable not in json_data["vars"]:
                    logger.warning(
                        f"[WARN] Model {filename} is rejected since the required variable '{required_variable}' was not in '{json_data['vars']}'"
                    )
                    continue

            coords = list()
            coords.append(str(json_data["geospatial_lon_min"]))
            coords.append(str(json_data["geospatial_lon_max"]))
            coords.append(str(json_data["geospatial_lat_min"]))
            coords.append(str(json_data["geospatial_lat_max"]))

            # Cesium takes depth in meters
            if standard_units(json_data["geospatial_vertical_units"]) == "m":
                depth_factor = 1000
            elif standard_units(json_data["geospatial_vertical_units"]) == "km":
                depth_factor = 1
            else:
                depth_factor = 1
                logger.error(
                    f"[ERR] Invalid depth unit of {json_data['geospatial_vertical_units']}"
                )

            coords.append(
                str(float(json_data["geospatial_vertical_min"]) / depth_factor)
            )
            coords.append(
                str(float(json_data["geospatial_vertical_max"]) / depth_factor)
            )

            model_coords = f"({','.join(coords)})"
            if "title" in json_data:
                title_list.append(json_data["title"])
            else:
                title_list.append("-")
            model_name = json_data["model"]
            coords_list.append(model_coords)
            model_list.append(model_name)

    # Prepare the HTML for the dropdown
    dropdown_html = f'<option value="">None</option>'
    for i, filename in enumerate(model_list):
        selected = " selected" if i == 0 else ""
        dropdown_html += (
            f'<option value="{coords_list[i]}"{selected}>{model_list[i]}</option>'
        )
        logger.debug(f"dropdown_html {dropdown_html}")
    return dropdown_html


# Route to create html for model dropdown. A simple drop-down of the file names and model variables.
@router.get("/models_drop_down", response_class=HTMLResponse)
def models_drop_down(required_variable: Optional[str] = None):
    logger.debug(f"Received route_name: {required_variable}")
    # init_model = "Cascadia-ANT+RF-Delph2018.r0.1.nc"
    json_directory = JSON_DIR
    netcdf_directory = NETCDF_DIR
    file_list = list()
    vars_list = list()
    model_list = list()
    for filename in sorted(os.listdir(json_directory)):
        nc_filename = filename.replace(".json", ".nc")
        filepath = os.path.join(netcdf_directory, nc_filename)

        if os.path.isfile(filepath):

            # Opening the file and loading the data
            with open(os.path.join(json_directory, filename), "r") as file:
                json_data = json.load(file)
                # Skip the file if file does not have the require variable
                if required_variable is not None:
                    if required_variable not in json_data["vars"]:
                        logger.warning(
                            f"[WARN] Model {filename} is rejected since the required variable '{required_variable}' was not in '{json_data['vars']}'"
                        )
                        continue
                file_list.append(nc_filename)
                data_vars = f"({','.join(json_data['data_vars'])})"
                model_name = json_data["model"]
                vars_list.append(data_vars)
                model_list.append(model_name)

    # Prepare the HTML for the dropdown
    dropdown_html = ""
    for i, filename in enumerate(file_list):
        selected = " selected" if i == 0 else ""
        dropdown_html += (
            f'<option value="{filename}"{selected}>{model_list[i]}</option>'
        )
    logger.debug(f"dropdown_html {dropdown_html}")
    return dropdown_html

@router.get("/get-token")
async def get_cesium_key(request: Request):
    """This must be secured lated to avoid external"""

    # Only internal requests are accepted.
    if os.getenv("CESIUM_KEYS") is None:
        logging.warning(f"[INFO] get-token Failed")
        return {"token": "your_access_token"}
    else:
        CESIUM_KEYS = json.loads(os.getenv("CESIUM_KEYS"))
        access_token = CESIUM_KEYS["cesium_access_token"]
        return {"token": access_token}

# Route to display the table of JSON files with auxiliary information hidden and filtering options.
@router.get("/list_filtered_json_files/{selected_model}", response_class=HTMLResponse)
async def list_table(request: Request, selected_model: str):
    # Collect all available variables from the JSON files
    all_variables = set()
    directory = JSON_DIR
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".json"):
            with open(filepath, "r") as file:
                data = json.load(file)
                data_vars = data.get("data_vars", [])
                all_variables.update(data_vars)
    all_variables = sorted(all_variables)

    # Generate the HTML form with sliders and multiple-select dropdown
    html_content = f"""
    <form id="filterForm">
        <label for="lat_min_filter">Latitude Min:</label>
        <input type="range" name="lat_min_filter" id="lat_min_filter" min="-90" max="90" step="0.1" value="-90" oninput="document.getElementById('lat_min_value').innerText = this.value"><span id="lat_min_value">-90</span><br>
        <label for="lat_max_filter">Latitude Max:</label>
        <input type="range" name="lat_max_filter" id="lat_max_filter" min="-90" max="90" step="0.1" value="90" oninput="document.getElementById('lat_max_value').innerText = this.value"><span id="lat_max_value">90</span><br>
        <label for="lon_min_filter">Longitude Min:</label>
        <input type="range" name="lon_min_filter" id="lon_min_filter" min="-180" max="180" step="0.1" value="-180" oninput="document.getElementById('lon_min_value').innerText = this.value"><span id="lon_min_value">-180</span><br>
        <label for="lon_max_filter">Longitude Max:</label>
        <input type="range" name="lon_max_filter" id="lon_max_filter" min="-180" max="180" step="0.1" value="180" oninput="document.getElementById('lon_max_value').innerText = this.value"><span id="lon_max_value">180</span><br>
        <label for="variables_filter">Variables:</label>
        <select name="variables_filter" id="variables_filter" multiple>
    """

    for variable in all_variables:
        html_content += f'<option value="{variable}">{variable}</option>'

    html_content += f"""
        </select><br>
        <input type="submit" value="Filter">
        <input type="reset" value="Clear" onclick="window.location.href='/list_filtered_json_files/{selected_model}'">
    </form>
    <div id="tableContainer">
    """

    html_content += await generate_table(selected_model)

    html_content += f"""
    </div>
    <script>
    document.getElementById('filterForm').addEventListener('submit', function(event) {{
        event.preventDefault();
        const formData = new FormData(event.target);
        fetch('/list_filtered_json_files/{selected_model}', {{
            method: 'POST',
            body: formData
        }}).then(response => response.text())
        .then(html => {{
            document.getElementById('tableContainer').innerHTML = html;
        }});
    }});
    </script>
    """

    return HTMLResponse(content=html_content)


# Route to display the table of JSON files with some data hidden.
@router.get("/list_json_files/{selected_model}", response_class=HTMLResponse)
def list_table(request: Request, selected_model: str):
    header_style = (
        "background-color: rgba(0, 79, 89, 0.8); color:white;font-size:10pt;"
    )
    checkbox_style = (
        "background-color: rgba(0, 79, 89, 0.8); color:white; font-size:10pt; border: none; padding: 5px;"
    )

    # Add a checkbox for toggling visibility.
    html_content = f"""
    <div id='tableContainer' style='max-height: 600px; overflow-y: auto; border: 1px solid #ccc; margin-top: 10px;'>
    <table style='width: 100%; border-collapse: collapse;'>
    <tr style='{header_style}'>
    <th></th>
    <th>3D Model</th>
    <th>Summary</th>
    <th class='hidden'>lat_min</th>
    <th class='hidden'>lat_max</th>
    <th class='hidden'>lon_min</th>
    <th class='hidden'>lon_max</th>
    <th class='hidden'>file</th>
    <th>Variable(s)</th>
    <th class='hidden'>depth_min</th>
    <th class='hidden'>depth_max</th>
    <th>Description</th>
    </tr>
    """

    directory = JSON_DIR
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(".json"):
            with open(filepath, "r") as file:
                data = json.load(file)
                lon_min = data.get("geospatial_lon_min", "-")
                lon_max = data.get("geospatial_lon_max", "-")
                lat_min = data.get("geospatial_lat_min", "-")
                lat_max = data.get("geospatial_lat_max", "-")
                depth_min = data.get("geospatial_vertical_min", "-")
                depth_max = data.get("geospatial_vertical_max", "-")
                model = data.get("model", "-")
                summary = data.get("summary", "-")
                filename_no_extension = os.path.splitext(filename)[0]
                data_vars = data.get("data_vars", "")
                var_desc = list()
                for var in data_vars:
                    desc = data.get("variables", "-")
                    if desc == "-":
                        var_desc.append(var)
                    elif var in desc:
                        if "display_name" in desc[var]:
                            var_desc.append(desc[var]["display_name"])
                        elif "long_name" in desc[var] and "units" in desc[var]:
                            var_desc.append(
                                f'{desc[var]["long_name"]} ({desc[var]["units"]})'
                            )
                        elif "long_name" in desc[var]:
                            var_desc.append(f'{desc[var]["long_name"]}')
                        else:
                            var_desc.append(var)
                zvar = data.get("z_var", "-")

                # Check if the current model matches the selected_model
                row_style = (
                    "background-color: rgba(0, 79, 89, 0.4);"
                    if model == selected_model
                    else ""
                )

                # Define the parameters
                params = {"filename": filename}

                # Encode the parameters
                encoded_params = urlencode(params)

                # Construct the full URL
                base_url = "/data/download-netcdf/"
                full_url = f"{base_url}?{encoded_params}"

                size_kb = data.get("size_kb", "")
                if size_kb:
                    size_kb = f", netCDF {size_kb:,} KB"

                html_content += f"""
                <tr style='{row_style}'>
                <td><input type='radio' name='selection' {'checked' if model == selected_model else ''} style="display:none;"></td>
                <td style='text-decoration:underline;cursor:pointer;font-size:8pt;'>{model}</td>
                <td style='font-size:9pt;'>{summary} <a href='{full_url}' download>download</a>{size_kb}</td>
                <td class='hidden'>{lat_min}</td>
                <td class='hidden'>{lat_max}</td>
                <td class='hidden'>{lon_min}</td>     
                <td class='hidden'>{lon_max}</td>   
                <td class='hidden' name='filename'>{filename_no_extension}</td>
                <td style='font-size:9pt;'><b>{", ".join(data_vars)}</b><br /><i>f({zvar})</i></td>
                <td class='hidden'>{depth_min}</td>  
                <td class='hidden'>{depth_max}</td>    
                <td style='font-size:9pt;'>{", ".join(var_desc)}</td>
                </tr>
                """

    html_content += "</table></div>"


    return html_content



def project_lonlat_utm(
    longitude, latitude, utm_zone, ellipsoid, xy_to_latlon=False, preserve_units=False
):
    """
    Performs cartographic transformations. Converts from longitude, latitude to UTM x,y coordinates
    and vice versa using PROJ (https://proj.org).

     Keyword arguments:
    longitude (scalar or array) – Input longitude coordinate(s).
    latitude (scalar or array) – Input latitude coordinate(s).
    xy_to_latlon (bool, default=False) – If inverse is True the inverse transformation from x/y to lon/lat is performed.
    preserve_units (bool) – If false, will ensure +units=m.
    """
    P = Proj(
        proj="utm",
        zone=utm_zone,
        ellps=ellipsoid,
    )
    # preserve_units=preserve_units,

    x, y = P(
        longitude,
        latitude,
        inverse=xy_to_latlon,
    )
    return x, y


def read_image(image_path):
    """Read an image file."""
    try:
        return plt.imread(image_path)
    except Exception as e:
        logger.error(f"[ERR] Failed to read image {image_path}: {e}")
        raise


def get_colormap_names():
    """Returns a list of all colormap names in matplotlib."""
    continuous_colormaps = []
    discrete_colormaps = []

    for cmap_name in plt.colormaps():
        cmap = plt.get_cmap(cmap_name)
        if isinstance(cmap, mcolors.ListedColormap):
            discrete_colormaps.append(cmap_name)
        else:
            continuous_colormaps.append(cmap_name)

    return continuous_colormaps, discrete_colormaps


@router.get("/colors", response_class=HTMLResponse)
async def color_dropdown():
    default = "red"
    colors = sorted(["red", "green", "blue", "yellow", "orange","cyan", "magenta", "purple", "brown", "black"])
    options_html = ""

    # Add continuous colors
    for color in colors:
        selected = " selected" if color == default else ""
        options_html += f'<option value="{color}"{selected}>{color}</option>'

    html_content = f"""
        <select name="color">
            {options_html}
        </select>"""
    return html_content

@router.get("/colormaps", response_class=HTMLResponse)
async def colormap_dropdown():
    default = "jet_r"
    continuous_colormaps, discrete_colormaps = get_colormap_names()
    options_html = ""

    # Add continuous colormaps
    options_html += '<optgroup label="Continuous colormaps">'
    for cmap in continuous_colormaps:
        selected = " selected" if cmap == default else ""
        options_html += f'<option value="{cmap}"{selected}>{cmap}</option>'
    options_html += "</optgroup>"

    # Add discrete colormaps
    options_html += '<optgroup label="Discrete colormaps">'
    for cmap in discrete_colormaps:
        selected = " selected" if cmap == default else ""
        options_html += f'<option value="{cmap}"{selected}>{cmap}</option>'
    options_html += "</optgroup>"

    html_content = f"""
        <select name="colormap">
            {options_html}
        </select>"""
    return html_content


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


# Plot an interpolated depth-profiles.
@router.route("/profiles", methods=["POST"])
async def depth_profiles(request: Request):
    version = "v.2024.102"
    version_label = f"{version}  {utc_now()['date_time']} UTC"
    form_data = await request.form()

    items_dict = dict(form_data.items())
    logger.debug(f"POST: {items_dict}")

    try:
        # lazy loading with Dask.
        plot_data = xr.open_dataset(
            os.path.join("static", "netcdf", items_dict["data_file"]), chunks={}
        )
        plot_var = items_dict["plot_variable"]
        start = [float(items_dict["start_lat"]), float(items_dict["start_lng"])]
        depth = [float(items_dict["start_depth"]), float(items_dict["end_depth"])]
        color = items_dict["color"]

        units = items_dict["units"]
        title = items_dict["title"]
        image_width = float(items_dict["image_width"])
        image_height = float(items_dict["image_height"])
        figsize = (image_width, image_height)
        logger.debug(
            f"converting the depth units {plot_data['depth']},\nunits: {plot_data['depth'].attrs['units']}"
        )
        
        # Validate depth unit conversion
        depth_unit_standard, depth_factor = unit_conversion_factor(
            plot_data["depth"].attrs["units"], units
        )
        if depth_factor is None:
            logger.error(f"Invalid depth conversion factor: {depth_factor}")
            return Response(
                content=create_error_image(f"Invalid depth conversion factor for units {units}"),
                media_type="image/png",
            )

        # Variable's unit conversion
        var_unit_standard, var_factor = unit_conversion_factor(
            plot_data[plot_var].attrs["units"], units
        )
        if var_factor is None:
            logger.error(f"Invalid variable conversion factor: {var_factor}")
            return Response(
                content=create_error_image(f"Invalid variable conversion factor for units {units}"),
                media_type="image/png",
            )
        
        logger.debug(
            f"depth units: {plot_data['depth'].attrs['units']},  {depth_factor}"
        )
        plot_data["depth"].attrs["units"] = depth_unit_standard
        plot_data[plot_var].attrs["units"] = var_unit_standard
        
        x_label = f"{plot_data[plot_var].attrs['long_name']} ({var_unit_standard})"
        y_label = f"{plot_data['depth'].attrs['long_name']} ({depth_unit_standard})"

        plot_data["depth"] = plot_data["depth"] * depth_factor
        plot_data["depth"].attrs["units"] = units

        plot_data = plot_data.where(
            (plot_data.depth >= float(depth[0])) & (plot_data.depth <= float(depth[1])),
            drop=True,
        ) * var_factor
    except Exception as ex:
        logger.error(f"[ERR] {ex}", exc_info=True)
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.format_exc()}"
            ),
            media_type="image/png",
        )

    utm_zone = None
    meta = plot_data.attrs
    dpi = 100
    fig_width, fig_height = figsize  # Figure size in inches
    logger.debug(f"Figure size: {fig_width}, {fig_height}")
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    logger.debug(f"META: {meta}")
    if "grid_ref" not in meta:
        logger.warning(
            f"[WARN] The 'grid_ref' attribute not found. Assuming geographic coordinate system"
        )
        grid_ref = "latitude_longitude"
    else:
        grid_ref = meta["grid_ref"]

    # Cross-section interpolation type.
    interp_type = items_dict["interpolation_method"]

    try:
        plot_data = extract_values(
            plot_data,
            start[0],
            start[1],
            depth,
            grid_ref = grid_ref,
            interpolation_method=interp_type,
            utm_zone=meta["utm_zone"],
            ellipsoid=meta["ellipsoid"],
        )
    except Exception as ex:
        message = f"[ERR] cross_section failed: {ex}\n{traceback.format_exc()}"
        logger.error(message)
        return Response(content=create_error_image(message), media_type="image/png")

    logger.debug(f"plot_data after cross_section: plot_data:{plot_data[plot_var].attrs}")
    logger.debug(f"plot_data values of {plot_var} after cross_section: plot_data:{plot_data[plot_var].values}")

    data_to_plot = plot_data[plot_var]

    logger.info(f"[INFO] Cross-section input: {items_dict}")
    data_to_plot["depth"] = data_to_plot["depth"] * -1
    logger.debug(f"data_to_plot:{data_to_plot}")
    
    try:
        # Extract the x-axis (distance) and y-axis (depth) values
        x_values = data_to_plot.values
        y_values = data_to_plot['depth'].values
        logger.debug(f"x_values: {x_values}")
        logger.debug(f"y_values: {y_values}")

        data_values = data_to_plot.values  # Assuming data_to_plot is a 2D array-like structure

        plt.plot(x_values, y_values, linestyle="--", marker="o", color=color)
        # Invert the y-axis (depth increasing downward)
        plt.gca().set_aspect("auto")
        plt.gca().invert_yaxis()

        plt.title(title)

    except Exception as ex:
        message = f"[ERR] Bad data selection: {ex}\n{traceback.format_exc()}"
        logger.error(message)
        return Response(content=create_error_image(message), media_type="image/png")

    # Set the depth limits for display.
    if depth[1] <= depth[0]:
        logger.error(f"Invalid depth range: {depth}")
        return Response(
            content=create_error_image(f"Invalid depth range: {depth}"),
            media_type="image/png",
        )
        
    plt.ylim(-depth[1], -depth[0])

    # Handle custom y-axis labels
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    y_ticks = plt.gca().get_yticks()

    # Assuming the labels are numeric, convert them to float, multiply by -1, and set them back.
    new_labels = []
    for label in labels:
        try:
            new_labels.append(custom_formatter(-1.0 * float(label.replace("−", "-"))))
        except ValueError:
            new_labels.append(label)  # If not numeric, keep the label as-is

    plt.gca().set_yticks(y_ticks)
    plt.gca().set_yticklabels(new_labels)

    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Adjust layout to make space for the legend
    plt.subplots_adjust(bottom=0.3)

    fig.tight_layout()

    # Save plot to BytesIO buffer and encode in Base64
    plot_buf = BytesIO()
    plt.savefig(plot_buf, format="png", bbox_inches="tight")
    plot_buf.seek(0)
    base64_plot = base64.b64encode(plot_buf.getvalue()).decode("utf-8")
    plt.close(fig)  # Properly close the figure to free memory

    # Convert xarray dataset to CSV (for simplicity in this example)
    csv_buf = BytesIO()
    plot_data[plot_var].to_dataframe().to_csv(csv_buf)
    csv_buf.seek(0)
    base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    content = {"image": base64_plot, "csv_data": base64_csv}

    plot_data.close()
    del plot_data, data_to_plot
    gc.collect()

    # Return both the plot and data as Base64-encoded strings
    return JSONResponse(content=content)

# Plot an interpolated cross-section.
@router.route("/xsection3d", methods=["POST"])
async def xsection(request: Request):
    version = "v.2024.102"
    version_label = f"{version}  {utc_now()['date_time']} UTC"
    form_data = await request.form()

    items_dict = dict(form_data.items())
    logger.debug(f"POST: {items_dict}")

    try:
        # lazy loading with Dask.
        plot_data = xr.open_dataset(
            os.path.join("static", "netcdf", items_dict["data_file"]), chunks={}
        )
        vertical_exaggeration = float(items_dict["vertical_exaggeration"])

        start = [float(items_dict["start_lat"]), float(items_dict["start_lng"])]
        end = [float(items_dict["end_lat"]), float(items_dict["end_lng"])]
        depth = [float(items_dict["start_depth"]), float(items_dict["end_depth"])]

        units = items_dict["units"]
        title = items_dict["title"]
        image_width = float(items_dict["image_width"])
        image_height = float(items_dict["image_height"])
        figsize = (image_width, image_height)
        logger.debug(
            f"converting the depth units {plot_data['depth']},\nunits: {plot_data['depth'].attrs['units']}"
        )
        unit_standard, depth_factor = unit_conversion_factor(
            plot_data["depth"].attrs["units"], units
        )
        logger.debug(
            f"depth units: {plot_data['depth'].attrs['units']},  {depth_factor}"
        )
        plot_data["depth"].attrs["units"] = unit_standard
        x_label = f"distance ({unit_standard})"
        y_label = f"{plot_data['depth'].attrs['long_name']} ({unit_standard}, VE: {vertical_exaggeration}x )"

        plot_data["depth"] = plot_data["depth"] * depth_factor
        plot_data["depth"].attrs["units"] = units

        plot_data = plot_data.where(
            (plot_data.depth >= float(depth[0])) & (plot_data.depth <= float(depth[1])),
            drop=True,
        )
    except Exception as ex:
        logger.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )
    utm_zone = None
    meta = plot_data.attrs
    dpi = 100
    fig_width, fig_height = figsize  # Figure size in inches
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    logger.debug(f"META: {meta}")
    if "grid_ref" not in meta:
        logger.warning(
            f"[WARN] The 'grid_ref' attribute not found. Assuming geographic coordinate system"
        )
        grid_ref = "latitude_longitude"
    else:
        grid_ref = meta["grid_ref"]

        # Cross-section interpolation type.
        interp_type = items_dict["interpolation_method"]

        # Steps in the cross-section.
        steps = int(items_dict["num_points"])
        logger.info(
            f"[INFO] cross_section start:{start}, end: {end}, steps: {steps}, interp_type: {interp_type}, plot_data: {plot_data}"
        )
        # Extract the cross-section.
        logger.info(
            f"[INFO] Before cross_section (plot_data,start,end,steps,interp_type:)\n{plot_data},{start},{end},{steps},{interp_type}"
        )
        try:
            plot_data, latitudes, longitudes = interpolate_path(
                plot_data,
                start,
                end,
                num_points=steps,
                method=interp_type,
                grid_ref=grid_ref,
                utm_zone=meta["utm_zone"],
                ellipsoid=meta["ellipsoid"],
            )

        except Exception as ex:
            message = f"[ERR] cross_section failed: {ex}\n{traceback.print_exc()}"
            logger.error(message)
            return Response(content=create_error_image(message), media_type="image/png")

        plot_var = items_dict["plot_variable"]

        logger.debug(f"After cross_section: plot_data:{plot_data[plot_var]}")
        logger.debug(f"After cross_section: plot_data:{plot_data[plot_var].values}")
        # Extract latitude and longitude from your cross-section data
        # latitudes = plot_data['latitude'].values
        # longitudes = plot_data['longitude'].values

        # If the original is not geographic, going outside the model coverage will result in NaN values.
        # Recompute these using the primary coordinates.
        # for _index, _lon in enumerate(plot_data['latitude'].values):
        #     if  np.isnan(_lon):
        #         plot_data['longitude'].values[_index], plot_data['latitude'].values[_index] = project_lonlat_utm(
        #         plot_data['x'].values[_index], plot_data['y'].values[_index], utm_zone, meta["ellipsoid"], xy_to_latlon=True)

        # Geod object for WGS84 (a commonly used Earth model)
        geod = Geod(ellps="WGS84")

        # Calculate distances between consecutive points
        _, _, distances = geod.inv(
            longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:]
        )

        # Compute cumulative distance, starting from 0
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

        if units == "mks":
            cumulative_distances = cumulative_distances / 1000.0

        logger.debug(f"units: {units}, cumulative_distances: {cumulative_distances}")

        # Assuming 'plot_data' is an xarray Dataset or DataArray
        # Create a new coordinate 'distance' based on the cumulative distances
        plot_data = plot_data.assign_coords(distance=("points", cumulative_distances))

        # If you want to use 'distance' as a dimension instead of 'index',
        # you can swap the dimensions (assuming 'index' is your current dimension)
        plot_data = plot_data.swap_dims({"points": "distance"})
        logger.debug(f"plot_data:{plot_data}")
        data_to_plot = plot_data[plot_var]

        # Iterate through the model variables and plot each cross-section.
        cmap = items_dict["colormap"]

        # plot_data["depth"] = -plot_data["depth"]

        vmin = items_dict["start_value"].strip()
        if vmin == "auto":
            vmin = ""

        vmax = items_dict["end_value"].strip()
        if vmax == "auto":
            vmax = ""

        logger.info(
            f"[INFO] plot_var units: {data_to_plot.attrs['units']}, units: {units}"
        )
        unit_standard, var_factor = unit_conversion_factor(
            data_to_plot.attrs["units"], units
        )
        logger.debug(
            f"plot_var units: {data_to_plot.attrs['units']}, units: {units} => unit_standard:{unit_standard}, var_factor: {var_factor}"
        )
        data_to_plot = data_to_plot * var_factor

        data_to_plot.attrs["units"] = unit_standard

        logger.info(f"[INFO] Cross-section input: {items_dict}")
        # logger.info(f"[INFO] Cross-section input: {items_dict['data_file']}, start: {start}, end: {end}, step: {steps}, interp_type: {interp_type}, plot_var: {plot_var}")
        data_to_plot["depth"] = data_to_plot["depth"] * -1
        logger.debug(f"data_to_plot:{data_to_plot}")
        try:
            if vmin and vmax:
                vmin = min(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )
                vmax = max(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )

                data_to_plot.plot.contourf(
                    x="distance",
                    y="depth",
                    cmap=cmap,
                    vmin=float(vmin),
                    vmax=float(vmax),
                )
            elif vmin:
                data_to_plot.plot.contourf(
                    x="distance", y="depth", cmap=cmap, vmin=float(vmin)
                )
            elif vmax:
                data_to_plot.plot.contourf(
                    x="distance", y="depth", cmap=cmap, vmax=float(vmax)
                )
            else:
                data_to_plot.plot.contourf(cmap=cmap)
                # data_to_plot.plot.contourf(x="distance", y="depth", cmap=cmap)
        except Exception as ex:
            message = f"[ERR] Bad data selection: {ex}\n{traceback.print_exc()}"
            logger.error(message)
            return Response(content=create_error_image(message), media_type="image/png")

        # Set the depth limits for display.
        logger.warning(f"Depth limits:{depth}")
        plt.ylim(-depth[1], -depth[0])

        # plt.gca().invert_yaxis()  # Invert the y-axis to show depth increasing downwards
        # Getting current y-axis tick labels
        labels = [item.get_text() for item in plt.gca().get_yticklabels()]
        y_ticks = plt.gca().get_yticks()

        # Assuming the labels are numeric, convert them to float, multiply by -1, and set them back.
        # If labels are not set or are custom, you might need to adjust this part.
        new_labels = [
            custom_formatter(-1.0 * float(label.replace("−", "-"))) if label else 0.0
            for label in labels
        ]  # Handles empty labels as well

        # Setting new labels ( better to explicitly set both the locations of the ticks and their labels using
        # set_yticks along with set_yticklabels.)
        plt.gca().set_yticks(y_ticks)
        plt.gca().set_yticklabels(new_labels)

        plt.ylabel(y_label)
        plt.xlabel(x_label)

        # Plotting using the current axis
        fig = plt.gcf()  # Get current figure, if you already have a figure created
        ax = plt.gca()  # Get current axes

        # Adding vertical text for start and end locations
        plt.text(
            cumulative_distances[0],
            1.05,
            f"⟸{start}",
            rotation=90,
            transform=plt.gca().get_xaxis_transform(),
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=9,
        )
        plt.text(
            cumulative_distances[-1],
            1.05,
            f"⟸{end}",
            rotation=90,
            transform=plt.gca().get_xaxis_transform(),
            verticalalignment="bottom",
            horizontalalignment="center",
            fontsize=9,
        )

        # Adjust layout to make space for the legend
        plt.subplots_adjust(bottom=0.3)

        # Setting the aspect ratio to 1:1 ('equal')
        plt.gca().set_aspect(vertical_exaggeration)
        fig.tight_layout()

        # Assuming `plt` is already being used for plotting
        fig = plt.gcf()  # Get the current figure
        axes = fig.axes  # Get all axes objects in the figure

        # The first axes object (`axes[0]`) is typically the main plot
        # The last axes object (`axes[-1]`) is likely the colorbar, especially if it was added automatically
        plot_axes = axes[0]
        colorbar_axes = axes[-1]
        colorbar_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # Get the position of the plot's y-axis
        plot_pos = plot_axes.get_position()

        # Adjust the colorbar's position
        # The numbers here are [left, bottom, width, height]
        # You may need to adjust these values based on your specific figure layout
        new_cbar_pos = [plot_pos.x1 + 0.01, plot_pos.y0, 0.02, plot_pos.height]
        colorbar_axes.set_position(new_cbar_pos)

        plt.title(title)

        ax = axes[0]
        # Logo for the plot.
        if os.path.isfile(LOGO_FILE):

            plot_ax = axes[0]
            logo = plt.imread(LOGO_FILE)
            # Get aspect ratio of the logo to maintain proportion
            aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height
            # Get aspect ratio of the logo to maintain proportion
            aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height

            # Desired logo height in inches and its normalized figure coordinates
            logo_height_inches = 0.3  # Height of the logo in inches
            logo_width_inches = (
                logo_height_inches * aspect_ratio
            )  # Calculate width based on aspect ratio

            # Normalize logo size relative to figure size
            norm_logo_width = logo_width_inches / fig_width
            norm_logo_height = logo_height_inches / fig_height

            # Calculate logo position in normalized coordinates
            norm_offset_below = LOGO_INCH_OFFSET_BELOW / fig_height

            # Position of logo's bottom edge
            logo_bottom = (
                plot_ax.get_position().ymin - norm_logo_height - norm_offset_below
            )
            logo_left = plot_ax.get_position().xmin  # Align left edges

            # Create an axes for the logo at the calculated position
            logo_ax = fig.add_axes(
                [logo_left, logo_bottom, norm_logo_width, norm_logo_height]
            )
            logo_ax.imshow(logo)
            logo_ax.axis("off")  # Hide the axis

            # Add a line of text below the logo
            text_y_position = -logo_bottom - norm_logo_height
            logo_ax.text(
                logo_left,
                text_y_position,
                f"{version_label}\n",
                ha="left",
                fontsize=6,
                color="#004F59",
            )

        # Save plot to BytesIO buffer and encode in Base64
        plot_buf = BytesIO()
        plt.savefig(plot_buf, format="png", bbox_inches="tight")
        plot_buf.seek(0)
        base64_plot = base64.b64encode(plot_buf.getvalue()).decode("utf-8")
        plt.clf()

    # Convert xarray dataset to CSV (for simplicity in this example)
    # You can also consider other formats like NetCDF for more complex data
    csv_buf = BytesIO()
    plot_data[plot_var].to_dataframe().to_csv(csv_buf)
    csv_buf.seek(0)
    base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    content = {"image": base64_plot, "csv_data": base64_csv}

    plot_data.close()
    del plot_data
    del data_to_plot
    gc.collect()

    # Return both the plot and data as Base64-encoded strings
    return JSONResponse(content=content)


# Plot an interpolated depth slice.
@router.route("/depth-slice-viewer", methods=["POST"])
async def depthslice(request: Request):
    form_data = await request.form()
    version = "v.2024.102"
    version_label = f"{version}  {utc_now()['date_time']} UTC"

    items_dict = dict(form_data.items())
    logger.info(f"[INFO] POST: {items_dict}")

    try:
        plot_data = xr.load_dataset(
            os.path.join("static", "netcdf", items_dict["data_file"])
        )
        logger.debug(f"POST: {items_dict}")
        start_lat = float(items_dict["start_lat"])
        start_lon = float(items_dict["start_lng"])
        end_lat = float(items_dict["end_lat"])
        end_lon = float(items_dict["end_lng"])
        depth = float(items_dict["start_depth"])
        lon_points = int(items_dict["lng_points"])
        lat_points = int(items_dict["lat_points"])
        image_width = float(items_dict["image_width"])
        image_height = float(items_dict["image_height"])
        z_var = items_dict["z_var"]
        figsize = (image_width, image_height)
        dpi = 100

        plot_grid_ref = items_dict["plot_grid_ref"]
        units = items_dict["units"]
        title = items_dict["title"]

        if z_var == "depth":
            unit_standard, depth_factor = unit_conversion_factor(
                plot_data["depth"].attrs["units"], units
            )
            logger.debug(
                f"depth units: {plot_data['depth'].attrs['units']} to {unit_standard}, {depth_factor}"
            )
        else:
            depth_factor = 1
            unit_standard = "testunit"

        if depth_factor != 1:
            # Create a new array with modified depth values
            new_depth_values = plot_data["depth"] * depth_factor

            # Create a new coordinate with the new values and the same attributes
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=plot_data["depth"].attrs
            )

            # Update to the new units.
            new_depth.attrs["units"] = unit_standard

            # Update to the new long name.
            if "standard_name" in new_depth.attrs:
                new_depth.attrs["long_name"] = (
                    f"{new_depth.attrs['standard_name']} [{unit_standard}]"
                )
            else:
                new_depth.attrs["long_name"] = (
                    f"{new_depth.attrs['variable']} [{unit_standard}]"
                )                

            # Replace the old 'depth' coordinate with the new one
            plot_data = plot_data.assign_coords(depth=new_depth)

        else:
            # Update to the new units.
            plot_data[z_var].attrs["units"] = unit_standard

            # Update to the new long name.
            if "standard_name" in plot_data[z_var].attrs:
                plot_data[z_var].attrs[
                    "long_name"
                ] = f"{plot_data[z_var].attrs['standard_name']} ({unit_standard})"
            else:
                plot_data[z_var].attrs[
                    "long_name"
                ] = f"{plot_data[z_var].attrs['variable']} ({unit_standard})"
        vmin = items_dict["start_value"].strip()
        if vmin == "auto":
            vmin = ""

        vmax = items_dict["end_value"].strip()
        if vmax == "auto":
            vmax = ""

        # We will be working with the variable dataset, so capture the metadata for the main dataset now.
        meta = plot_data.attrs

        # The plot variable.
        plot_var = items_dict["plot_variable"]

        # Interpolate the dataset vertically to the target depth
        plot_data = plot_data[plot_var]
        unit_standard, var_factor = unit_conversion_factor(
            plot_data.attrs["units"], units
        )
        plot_data.data *= var_factor
        logger.debug(f"{plot_data}\nAttrs: {plot_data.attrs}")
        plot_data.attrs["units"] = unit_standard
        plot_data.attrs["display_name"] = (
            f"{plot_data.attrs['long_name']} [{unit_standard}]"
        )

    except Exception as ex:
        logger.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )

    utm_zone = None
    logger.debug(f"META: {meta}")
    if "grid_ref" not in meta:
        logger.warning(
            f"[WARN] The 'grid_ref' attribute not found. Assuming geographic coordinate system"
        )
        grid_ref = "latitude_longitude"
    else:
        grid_ref = meta["grid_ref"]

    if "utm_zone" in meta:
        utm_zone = meta["utm_zone"]
        ellipsoid = meta["ellipsoid"]

        logger.info(f"[INFO] UTM zone: {utm_zone}")
        logger.info(f"[INFO] grid_ref: ", grid_ref)

    if grid_ref == "latitude_longitude":
        # Create a 2D grid of latitudes and longitudes
        lon_list = np.linspace(start_lon, end_lon, lon_points).tolist()
        lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
        x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
        x_grid, y_grid = np.meshgrid(lon_list, lat_list)
        logger.debug(
            f"plot_grid_ref: {plot_grid_ref}, grid_ref:{grid_ref}\nlat_list:{lat_list[0]} to {lat_list[-1]}, lon_list: {lon_list[0]} to {lon_list[-1]}"
        )

    else:
        start_x, start_y = project_lonlat_utm(start_lon, start_lat, utm_zone, ellipsoid)
        end_x, end_y = project_lonlat_utm(end_lon, end_lat, utm_zone, ellipsoid)
        x_list = np.linspace(start_x, end_x, lon_points).tolist()
        y_list = np.linspace(start_y, end_y, lat_points).tolist()
        lon_list, lat_list = project_lonlat_utm(
            x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
        )
        x_grid, y_grid = np.meshgrid(x_list, y_list)
        logger.debug(
            f"plot_ grid_ref: {plot_grid_ref}, grid_ref:{grid_ref}\nx_list:{x_list[0]} to {x_list[-1]}, y_list: {y_list[0]} to {y_list[-1]}"
        )

    # Iterate through the model variables and plot each cross-section.
    cmap = items_dict["colormap"]
    # Cross-section interpolation type.
    interp_type = items_dict["interpolation_method"]
    figsize = (image_width, image_height)
    interpolated_values_2d = np.zeros_like(
        x_grid
    )  # Initialize a 2D array for storing interpolated values

    # No interpolation.
    if interp_type == "none":
        logger.debug(f"plot_data: {plot_data}")
        start_lat = closest(plot_data["latitude"].data, start_lat)
        end_lat = closest(plot_data["latitude"].data, end_lat)
        start_lon = closest(plot_data["longitude"].data, start_lon)
        end_lon = closest(plot_data["longitude"].data, end_lon)
        depth_closest = closest(list(plot_data[z_var].data), depth)
        plot_data = plot_data.where(plot_data[z_var] == depth_closest, drop=True)
        plot_data = plot_data.where(
            (plot_data.latitude >= start_lat)
            & (plot_data.latitude <= end_lat)
            & (plot_data.longitude >= start_lon)
            & (plot_data.longitude <= end_lon),
            drop=True,
        )
        # Creating the contour plot
        # ds_var.plot(figsize=(7, 10), cmap=cmap)#, vmin=vmin, vmax=vmax)
        logger.debug(
            f"x2, y2 ({plot_grid_ref}<=>{grid_ref}):{grid_ref_dict[plot_grid_ref]['x2']}, {grid_ref_dict[plot_grid_ref]['y2']}"
        )

        if plot_grid_ref == grid_ref:
            x2 = grid_ref_dict[plot_grid_ref]["x"]
            y2 = grid_ref_dict[plot_grid_ref]["y"]
            logger.debug(f"x: {x2}")
        else:
            x2 = grid_ref_dict[plot_grid_ref]["x2"]
            y2 = grid_ref_dict[plot_grid_ref]["y2"]
            logger.debug(f"x2: {x2}")

        if plot_grid_ref == "latitude_longitude":
            logger.debug(f"{plot_grid_ref}")
            unit_standard, x_factor = unit_conversion_factor(
                plot_data[x2].attrs["units"], units
            )
            plot_data[x2].attrs["units"] = unit_standard
            plot_data[x2].attrs[
                "display_name"
            ] = f"{plot_data[x2].attrs['long_name']} [{unit_standard}]"

            unit_standard, y_factor = unit_conversion_factor(
                plot_data[y2].attrs["units"], units
            )
            plot_data[y2].attrs["units"] = unit_standard
            plot_data[y2].attrs[
                "display_name"
            ] = f"{plot_data[y2].attrs['long_name']} [{unit_standard}]"
        else:
            logger.debug(f"{plot_data}")
            unit_standard, x_factor = unit_conversion_factor(
                plot_data[x2].attrs["units"], units
            )
            logger.debug(
                f"{x2} plot_grid_ref: {plot_grid_ref}, {unit_standard}, {x_factor}"
            )

            logger.debug(f"plot_data[x2].attrs:{ plot_data[x2].attrs}")
            new_x_values = plot_data[x2] * x_factor
            new_x = xr.DataArray(
                new_x_values, dims=plot_data[x2].dims, attrs=plot_data[x2].attrs
            )
            new_x.attrs["units"] = unit_standard
            new_x.attrs["display_name"] = (
                f"{new_x.attrs['long_name']} [{unit_standard}]"
            )
            logger.debug(f"NEW plot_data[x2].attrs:{ new_x}")

            plot_data = plot_data.assign_coords(x2=new_x)
            logger.debug(f"Updated plot_data[x2].attrs:{ plot_data[x2]}")

            unit_standard, y_factor = unit_conversion_factor(
                plot_data[y2].attrs["units"], units
            )

            new_y_values = plot_data[y2] * y_factor
            new_y = xr.DataArray(
                new_y_values, dims=plot_data[y2].dims, attrs=plot_data[y2].attrs
            )
            new_y.attrs["units"] = unit_standard
            new_y.attrs["display_name"] = (
                f"{new_y.attrs['long_name']} [{unit_standard}]"
            )
            plot_data = plot_data.assign_coords(y2=new_y)

        logger.debug(f"plot_data:{plot_data}")
        # Geographic or projected coordinates?
        if plot_grid_ref == "latitude_longitude":
            # Calculate the correct aspect ratio
            lat_list = plot_data.latitude.values.flatten()
            lon_list = plot_data.longitude.values.flatten()
            mid_lat = sum(lat_list) / len(lat_list)  # Average latitude
            aspect_ratio = 1 / math.cos(math.radians(mid_lat))
        else:
            aspect_ratio = 1
        x_list = plot_data[x2].values.flatten()
        y_list = plot_data[y2].values.flatten()
        fig_width, fig_height = figsize  # Figure size in inches
        # fig_height = fig_width * aspect_ratio * (max(y_list) - min(y_list)) /  (max(x_list) - min(x_list))
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

        logger.debug(
            f"plot_data: {plot_data}\n\n{plot_data.var}\n\n{plot_data.variable}"
        )
        colorbar_label = f"{plot_data.attrs['long_name']} [{standard_units(plot_data.attrs['units'])}]"
        try:
            logger.debug(f"PLOT y2: {y2}\n{plot_data[y2].values}")
            # Creating the contour plot
            # fig = plt.figure(figsize=figsize)
            if plot_grid_ref == "latitude_longitude":
                ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

                if vmin and vmax:
                    vmin = min(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )
                    vmax = max(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )

                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                        cbar_kwargs={
                            "shrink": 0.8,
                            "pad": 0.08,
                            "label": colorbar_label,
                            "format": "{x:0.1f}".format,
                        },
                    )
                elif vmin:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmin=float(vmin),
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                        cbar_kwargs={
                            "shrink": 0.8,
                            "pad": 0.08,
                            "label": colorbar_label,
                            "format": "{x:0.1f}".format,
                        },
                    )
                elif vmax:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        vmax=float(vmax),
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                        cbar_kwargs={
                            "shrink": 0.8,
                            "pad": 0.08,
                            "label": colorbar_label,
                            "format": "{x:0.1f}".format,
                        },
                    )
                else:

                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
                        cbar_kwargs={
                            "shrink": 0.8,
                            "pad": 0.08,
                            "label": colorbar_label,
                            "format": "{x:0.1f}".format,
                        },
                    )

                plt.xticks(rotation=45)  # Rotating the x-axis labels to 45 degrees
                # Add coastlines
                ax.coastlines()

                # Optionally, add other geographic features
                ax.add_feature(cfeature.BORDERS, linestyle=":")

                # Add gridlines with labels only on the left and bottom
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False  # Disable top labels
                gl.right_labels = False  # Disable right labels
            else:
                if vmin and vmax:
                    vmin = min(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )
                    vmax = max(
                        float(items_dict["start_value"]), float(items_dict["end_value"])
                    )

                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax, x=x2, y=y2
                    )
                elif vmin:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmin=float(vmin), x=x2, y=y2
                    )
                elif vmax:
                    plot_data.plot(
                        figsize=figsize, cmap=cmap, vmax=float(vmax), x=x2, y=y2
                    )
                else:
                    plot_data.plot(figsize=figsize, cmap=cmap, x=x2, y=y2)

                plt.xticks(rotation=45)  # Rotating the x-axis labels to 45 degrees

        except Exception as ex:
            message = f"[ERR] Bad data selection: {ex}\n{traceback.print_exc()}"
            logger.error(message)
            return Response(content=create_error_image(message), media_type="image/png")
    # Interpolation.
    else:
        logger.info(
            f"[INFO] grid_ref: {grid_ref}, plot_grid_ref: {plot_grid_ref}"
        )

        # If you're working with longitude and latitude
        if grid_ref == "latitude_longitude":
            interpolated_values_2d = plot_data.interp(
                latitude=lat_list, longitude=lon_list, depth=depth, method=interp_type
            )
        else:
            interpolated_values_2d = plot_data.interp(
                y=y_list, x=x_list, depth=depth, method=interp_type
            )
        # Creating the contour plot
        # fig = plt.figure(figsize=figsize)

        # Check if all elements are NaN. No cubic spline with arrays with NaN.
        if np.all(np.isnan(interpolated_values_2d)) and interp_type == "cubic":
            message = (
                f"[ERR] Data with NaN values. Can't use the cubic spline interpolation"
            )
            logger.error(message)
            base64_plot = base64.b64encode(create_error_image(message)).decode("utf-8")
            plt.clf()

            # Convert xarray dataset to CSV (for simplicity in this example)
            # You can also consider other formats like NetCDF for more complex data
            csv_buf = BytesIO()
            plot_data.to_dataframe().to_csv(csv_buf)
            csv_buf.seek(0)
            base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

            content = {"image": base64_plot, "csv_data": base64_csv}

            plot_data.close()
            del plot_data
            gc.collect()

            # Return both the plot and data as Base64-encoded strings
            return JSONResponse(content=content)

        logger.info(f"[INFO] vmin, vmax (1): ", vmin, vmax)

        if grid_ref == plot_grid_ref:
            x_label = plot_data[grid_ref_dict[plot_grid_ref]["x"]].attrs[
                "long_name"
            ]
            y_label = plot_data[grid_ref_dict[plot_grid_ref]["y"]].attrs[
                "long_name"
            ]
        else:
            x_label = plot_data[grid_ref_dict[plot_grid_ref]["x2"]].attrs[
                "long_name"
            ]
            y_label = plot_data[grid_ref_dict[plot_grid_ref]["y2"]].attrs[
                "long_name"
            ]
        # Geographic or projected coordinates?
        if plot_grid_ref == "latitude_longitude":
            # Calculate the correct aspect ratio
            _list = plot_data.latitude.values.flatten()
            lon_range = max(plot_data.longitude.values.flatten()) - min(
                plot_data.longitude.values.flatten()
            )
            mid_lat = sum(_list) / len(_list)  # Average latitude
            aspect_ratio = 1 / math.cos(math.radians(mid_lat))
            fig_width, fig_height = figsize  # Figure size in inches
            # fig_height = fig_width * aspect_ratio * (max(_list) - min(_list)) / lon_range

        else:
            aspect_ratio = 1
            fig_width, fig_height = figsize  # Figure size in inches
            # fig_height = fig_width * aspect_ratio * (max(y_list) - min(y_list)) /  (max(x_list) - min(x_list))

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        if plot_grid_ref == "latitude_longitude":
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        logger.debug(f"figsize = {fig_width}, {fig_height}")

        try:
            if vmin and vmax:
                vmin = min(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )
                vmax = max(
                    float(items_dict["start_value"]), float(items_dict["end_value"])
                )
                # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap,vmin=vmin, vmax=vmax)
                # ds_var.plot(figsize=figsize, cmap=cmap,vmin=vmin, vmax=vmax)
            if not vmin:
                vmin = nanmin(interpolated_values_2d)
                # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap,vmin=vmin)
            if not vmax:
                vmax = nanmax(interpolated_values_2d)
                # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap, vmax=vmax)
            logger.info(f"[INFO] vmin, vmax (2): ", vmin, vmax)
            levels = np.linspace(vmin, vmax, 15)

            logger.debug(f"PLOT: plot_grid_ref: {plot_grid_ref}")
            if plot_grid_ref == "latitude_longitude":
                # contourf = plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=levels, cmap=cmap)#, x=grid_ref_dict[grid_ref]["x"], y=grid_ref_dict[grid_ref]["y"])
                # Plot the contourf with longitude and latitude lists
                contourf = ax.contourf(
                    lon_list,
                    lat_list,
                    interpolated_values_2d,
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                )

                # Add coastlines
                ax.coastlines()

                # Optionally, add other features like borders and gridlines
                ax.add_feature(cfeature.BORDERS)
                gl = ax.gridlines(draw_labels=True)
                gl.top_labels = False  # Disable top labels
                gl.right_labels = False  # Disable right labels
                # logger.warning(f"lon_list:{lon_list}\nlat_list:{lat_list}")

            else:
                logger.debug(f"y_list: {y_list}")
                contourf = plt.contourf(
                    x_list, y_list, interpolated_values_2d, levels=levels, cmap=cmap
                )  # , x=grid_ref_dict[grid_ref]["x"], y=grid_ref_dict[grid_ref]["y"])
                logger.debug(f"x_list:{x_list}\ny_list:{y_list}")
            logger.debug(f"Plot Limits = {plt.xlim()}, {plt.ylim()}")
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
        except Exception as ex:
            message = f"[ERR] Bad data selection: {ex}\n{traceback.print_exc()}"
            logger.error(message)
            return Response(content=create_error_image(message), media_type="image/png")
        # plt.contourf(lon_list, lat_list, interpolated_values_2d, levels=15, cmap=cmap)
        # plt.colorbar(label='Interpolated Value')
        cbar = plt.colorbar(contourf, ax=plt.gca())
        cbar.set_label(
            plot_data.attrs["display_name"]
        )  # Optionally add a label to the colorbar
    plt.xticks(rotation=45)  # Rotating the x-axis labels to 45 degrees

    # Get the dimensions of the Axes object in inches
    bbox = plt.gca().get_position()
    width, height = bbox.width * fig.get_figwidth(), bbox.height * fig.get_figheight()

    logger.debug(f"Axes width in inches: {width}")
    logger.debug(f"Axes height in inches: {height}")

    # fig.tight_layout()

    # Assuming `plt` is already being used for plotting
    fig = plt.gcf()  # Get the current figure
    axes = fig.axes  # Get all axes objects in the figure

    # The first axes object (`axes[0]`) is typically the main plot
    # The last axes object (`axes[-1]`) is likely the colorbar, especially if it was added automatically
    plot_axes = axes[0]
    colorbar_axes = axes[-1]
    # logger.warning(f"colorbar_label: {colorbar_label}")
    # logger.warning(f"colorbar_axes: {dir(colorbar_axes)}")
    # colorbar_axes.set_label(colorbar_label)
    # colorbar_axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Get the position of the plot's y-axis
    plot_pos = plot_axes.get_position()

    # Adjust the colorbar's position
    # The numbers here are [left, bottom, width, height]
    # You may need to adjust these values based on your specific figure layout
    new_cbar_pos = [plot_pos.x1 + 0.01, plot_pos.y0, 0.02, plot_pos.height]
    colorbar_axes.set_position(new_cbar_pos)
    colorbar_axes.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    plt.title(title)

    ax = axes[0]
    ax.set_aspect(aspect_ratio)  # Set the aspect ratio to the calculated value
    # Logo for the plot.
    if os.path.isfile(LOGO_FILE):

        plot_ax = axes[0]
        logo = plt.imread(LOGO_FILE)
        # Get aspect ratio of the logo to maintain proportion
        aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height
        # Get aspect ratio of the logo to maintain proportion
        aspect_ratio = logo.shape[1] / logo.shape[0]  # width / height

        # Desired logo height in inches and its normalized figure coordinates
        logo_height_inches = 0.3  # Height of the logo in inches
        logo_width_inches = (
            logo_height_inches * aspect_ratio
        )  # Calculate width based on aspect ratio

        # Normalize logo size relative to figure size
        norm_logo_width = logo_width_inches / fig_width
        norm_logo_height = logo_height_inches / fig_height

        # Calculate logo position in normalized coordinates
        norm_offset_below = 2 * LOGO_INCH_OFFSET_BELOW / fig_height

        # Position of logo's bottom edge
        logo_bottom = plot_ax.get_position().ymin - norm_logo_height - norm_offset_below
        logo_left = plot_ax.get_position().xmin  # Align left edges

        # Create an axes for the logo at the calculated position
        logo_ax = fig.add_axes(
            [logo_left, logo_bottom, norm_logo_width, norm_logo_height]
        )
        logo_ax.imshow(logo)
        logo_ax.axis("off")  # Hide the axis

        # Add a line of text below the logo
        text_y_position = -logo_bottom - norm_logo_height
        logo_ax.text(
            logo_left,
            text_y_position,
            f"{version_label}\n",
            ha="left",
            fontsize=6,
            color="#004F59",
        )

    else:
        logger.warning(f"[WARN] Logo file not found: {LOGO_FILE}")
    # Save the plot to a BytesIO object
    # plot_bytes = BytesIO()
    # plt.savefig(plot_bytes, format='png')
    # plot_bytes.seek(0)
    # plt.clf()
    # del ds_var
    # del ds
    # gc.collect()

    # Return the image as a response
    # return Response(content=plot_bytes.getvalue(), media_type="image/png")

    # Save plot to BytesIO buffer and encode in Base64
    plot_buf = BytesIO()
    plt.savefig(plot_buf, format="png", bbox_inches="tight")
    plot_buf.seek(0)
    base64_plot = base64.b64encode(plot_buf.getvalue()).decode("utf-8")
    plt.clf()
    # Convert xarray dataset to CSV (for simplicity in this example)
    # You can also consider other formats like NetCDF for more complex data
    """
    csv_buf = BytesIO()
    plot_data.to_dataframe().to_csv(csv_buf)
    csv_buf.seek(0)
    base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")
    content = {"image": base64_plot, "csv_data": base64_csv}
    """
    try:
        # Check if data is too large, for example by number of elements
        if plot_data.size > 1000000:  # Adjust this size limit as necessary
            raise ValueError("Data too large")

        # Convert data to DataFrame and then to CSV
        csv_buf = BytesIO()
        plot_data.to_dataframe().to_csv(csv_buf)
        csv_buf.seek(0)
        base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    except ValueError as e:
        # Handle cases where data is too large
        csv_buf = BytesIO()
        # Write a message indicating the data is too large
        csv_buf.write(b"Data too large")
        csv_buf.seek(0)
        base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    except Exception as e:
        # Handle other potential exceptions
        csv_buf = BytesIO()
        error_message = f"An error occurred: {str(e)}"
        csv_buf.write(error_message.encode())
        csv_buf.seek(0)
        base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")

    # Package the content
    content = {"image": base64_plot, "csv_data": base64_csv}

    plot_data.close()
    del plot_data
    gc.collect()

    # Return both the plot and data as Base64-encoded strings
    return JSONResponse(content=content)


@router.get("/slice-data")
async def slice_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    lng_points: int,
    lat_points: int,
    output_grid_ref: str,
    units: str,
    variables: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
    start_value: str = Query("auto"),
    end_value: str = Query("auto"),
    interpolation_method: str = Query("none"),
):
    """
    Generate an interpolated depth slice from a CVM model and return the data in the specified format.

    Args:
        request (Request): The request object.
        data_file (str): The path to the netCDF file.
        start_lat (float): The starting latitude.
        start_lng (float): The starting longitude.
        end_lat (float): The ending latitude.
        end_lng (float): The ending longitude.
        start_depth (float): The depth for the slice.
        lng_points (int): The number of longitude points.
        lat_points (int): The number of latitude points.
        output_grid_ref (str): The grid ref type (latitude_longitude or transverse_mercator).
        units (str): The units of measurement.
        variables (str): The variables to plot, comma-separated.
        output_format (str, optional): The output format (csv, xarray, netcdf, geocsv). Defaults to "csv".
        start_value (str, optional): The start value for the plot. Defaults to "auto".
        end_value (str, optional): The end value for the plot. Defaults to "auto".
        interpolation_method (str, optional): The interpolation method. Defaults to "none".

    Returns:
        Response: The response containing the requested data.
    """
    version = "v.2024.102"

    try:
        data_file = urllib.parse.unquote(data_file)  # Decode the URL-safe string
        output_data = xr.load_dataset(os.path.join("static", "netcdf", data_file))
        logger.info(f"[INFO] GET: {locals()}")

        unit_standard, depth_factor = unit_conversion_factor(
            output_data["depth"].attrs["units"], units
        )
        if depth_factor != 1:
            new_depth_values = output_data["depth"] * depth_factor
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=output_data["depth"].attrs
            )
            new_depth.attrs["units"] = unit_standard
            if "standard_name" in new_depth.attrs:
                new_depth.attrs["long_name"] = (
                    f"{new_depth.attrs['standard_name']} [{unit_standard}]"
                )
            else:
                new_depth.attrs["long_name"] = (
                    f"{new_depth.attrs['variable']} [{unit_standard}]"
                )
            output_data = output_data.assign_coords(depth=new_depth)
        else:
            output_data.depth.attrs["units"] = unit_standard
            if "standard_name" in output_data.depth.attrs:
                output_data.depth.attrs["long_name"] = (
                    f"{output_data.depth.attrs['standard_name']} ({unit_standard})"
                )
            else:
                utput_data.depth.attrs["long_name"] = (
                    f"{output_data.depth.attrs['variable']} ({unit_standard})"
                )

        vmin = None if start_value == "auto" else float(start_value)
        vmax = None if end_value == "auto" else float(end_value)

        meta = output_data.attrs

        variables_list = variables.split(",")  # Split the variables by comma
        selected_data_vars = output_data[variables_list]

        # Apply unit conversion to each variable
        for var in variables_list:
            unit_standard, var_factor = unit_conversion_factor(
                selected_data_vars[var].attrs["units"], units
            )
            selected_data_vars[var].data *= var_factor
            selected_data_vars[var].attrs["units"] = unit_standard
            selected_data_vars[var].attrs[
                "display_name"
            ] = f"{selected_data_vars[var].attrs['long_name']} [{unit_standard}]"

    except Exception as ex:
        logger.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )

    try:
        grid_ref = meta.get("grid_ref", "latitude_longitude")
        utm_zone = meta.get("utm_zone")
        ellipsoid = meta.get("ellipsoid")

        if grid_ref == "latitude_longitude":
            lon_list = np.linspace(start_lng, end_lng, lng_points).tolist()
            lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
            x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
            x_grid, y_grid = np.meshgrid(lon_list, lat_list)
        else:
            start_x, start_y = project_lonlat_utm(
                start_lng, start_lat, utm_zone, ellipsoid
            )
            end_x, end_y = project_lonlat_utm(end_lng, end_lat, utm_zone, ellipsoid)
            x_list = np.linspace(start_x, end_x, lng_points).tolist()
            y_list = np.linspace(start_y, end_y, lat_points).tolist()
            lon_list, lat_list = project_lonlat_utm(
                x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
            )
            x_grid, y_grid = np.meshgrid(x_list, y_list)

        interp_type = interpolation_method

        if interp_type == "none":
            start_lat = closest(output_data["latitude"].data, start_lat)
            end_lat = closest(output_data["latitude"].data, end_lat)
            start_lng = closest(output_data["longitude"].data, start_lng)
            end_lng = closest(output_data["longitude"].data, end_lng)
            depth_closest = closest(list(output_data["depth"].data), start_depth)
            selected_data_vars = selected_data_vars.where(
                output_data.depth == depth_closest, drop=True
            )
            selected_data_vars = selected_data_vars.where(
                (output_data.latitude >= start_lat)
                & (output_data.latitude <= end_lat)
                & (output_data.longitude >= start_lng)
                & (output_data.longitude <= end_lng),
                drop=True,
            )
            data_to_return = selected_data_vars
        else:
            interpolated_values_2d = {}
            for var in variables_list:
                if grid_ref == "latitude_longitude":
                    interpolated_values_2d[var] = selected_data_vars[var].interp(
                        latitude=lat_list,
                        longitude=lon_list,
                        depth=start_depth,
                        method=interp_type,
                    )
                else:
                    interpolated_values_2d[var] = selected_data_vars[var].interp(
                        y=y_list, x=x_list, depth=start_depth, method=interp_type
                    )

            if any(
                np.all(np.isnan(interpolated_values_2d[var])) and interp_type == "cubic"
                for var in variables_list
            ):
                message = "[ERR] Data with NaN values. Can't use the cubic spline interpolation"
                logger.error(message)
                base64_plot = base64.b64encode(create_error_image(message)).decode(
                    "utf-8"
                )
                plt.clf()
                csv_buf = BytesIO()
                selected_data_vars.to_dataframe().to_csv(csv_buf)
                csv_buf.seek(0)
                base64_csv = base64.b64encode(csv_buf.getvalue()).decode("utf-8")
                content = {"image": base64_plot, "csv_data": base64_csv}
                output_data.close()
                del output_data
                gc.collect()
                return JSONResponse(content=content)
            data_to_return = xr.Dataset(interpolated_values_2d)

        unique_id = uuid4().hex
        if output_format == "csv":
            df = data_to_return.to_dataframe().reset_index()
            csv_path = f"/tmp/plot_data_{unique_id}.csv"
            df.to_csv(csv_path, index=False)
            return FileResponse(
                path=csv_path, filename="plot_data.csv", media_type="text/csv"
            )
        elif output_format == "xarray":
            response_json = jsonable_encoder(data_to_return.to_dict())
            return JSONResponse(content=response_json)
        elif output_format == "netcdf":
            netcdf_path = f"/tmp/plot_data_{unique_id}.nc"
            data_to_return.to_netcdf(netcdf_path)
            return FileResponse(
                path=netcdf_path,
                filename="plot_data.nc",
                media_type="application/netcdf",
            )
        elif output_format == "geocsv":
            df = data_to_return.to_dataframe().reset_index()
            df.to_csv(csv_path, index=False)
            geocsv_path = f"/tmp/plot_data_{unique_id}.geocsv"
            df.to_csv(geocsv_path, index=False)
            return FileResponse(
                path=geocsv_path, filename="plot_data.geocsv", media_type="text/csv"
            )
        else:
            raise HTTPException(
                status_code=400, detail="Invalid output format specified"
            )

    except Exception as ex:
        logger.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(f"[ERR] {ex}\n{traceback.print_exc()}"),
            media_type="image/png",
        )


@router.post("/submitForm")
async def submit_form(
    name: str = Form(...),
    institution: str = Form(...),
    email: str = Form(...),
    message: str = Form(...),
):

    try:
        # Create SES client
        AWS_REGION = "us-east-2"
        SENDER_EMAIL = "manochehr.bahavar@earthscope.org"
        RECIPIENT_EMAIL = "manochehr.bahavar@earthscope.org"
        ses_client = boto3.client("ses", region_name=AWS_REGION)

        # Email content
        subject = "New Request from Form"
        body_text = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        body_html = f"""<html>
        <head></head>
        <body>
        <h1>New Request</h1>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Message:</strong> {message}</p>
        </body>
        </html>
                    """

        response = ses_client.send_email(
            Destination={
                "ToAddresses": [
                    RECIPIENT_EMAIL,
                ],
            },
            Message={
                "Body": {
                    "Html": {
                        "Charset": "UTF-8",
                        "Data": body_html,
                    },
                    "Text": {
                        "Charset": "UTF-8",
                        "Data": body_text,
                    },
                },
                "Subject": {
                    "Charset": "UTF-8",
                    "Data": subject,
                },
            },
            Source=SENDER_EMAIL,
        )
    except Exception as e:
        # Log the exception if needed
        logger.error(f"[ERR] Error sending email: {e}")
    return HTMLResponse(content="<p>Form submitted successfully!</p>", status_code=200)

    # return {"message": "Email sent successfully!"}
