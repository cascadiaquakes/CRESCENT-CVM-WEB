import os
import json
import traceback

from fastapi import APIRouter, Request, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
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

import math
import gc

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


@router.get("/test3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("test_3d.html", {"request": request})


@router.get("/volume-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("volume-3d.html", {"request": request})


@router.get("/div", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("div.html", {"request": request})


@router.get("/x-section", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("x-section.html", {"request": request})


# New Routes.
# Index page.
@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Cross-section viewer.
@router.get("/x-section-viewer", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("x-section-viewer.html", {"request": request})


# Depth-profile viewer.
@router.get("/depth-profile-viewer", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("depth-profile-viewer.html", {"request": request})

# Cross-section 3D map.
@router.get("/x-section-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("x-section-3d.html", {"request": request})


# Depth-profile 3D map.
@router.get("/depth-profile-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("depth-profile-3d.html", {"request": request})


# Depth-slice viewer.
@router.get("/depth-slice-viewer", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("depth-slice-viewer.html", {"request": request})


# Horizontal-slice viewer.
@router.get("/horizontal-slice-viewer", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse(
        "horizontal-slice-viewer.html", {"request": request}
    )


# Horizontal-slice 3D map.
@router.get("/horizontal-slice-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("horizontal-slice-3d.html", {"request": request})


# Depth-slice 3D map.
@router.get("/depth-slice-3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("depth-slice-3d.html", {"request": request})


# Route to create html for model dropdown.
@router.get("/models_drop_down", response_class=HTMLResponse)
def models_drop_down():
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
            file_list.append(nc_filename)
            # Opening the file and loading the data
            with open(os.path.join(json_directory, filename), "r") as file:
                json_data = json.load(file)
                data_vars = f"({','.join(json_data['data_vars'])})"
                model_name = json_data["model"]
                vars_list.append(data_vars)
                model_list.append(model_name)

    # Prepare the HTML for the dropdown
    dropdown_html = ""
    for i, filename in enumerate(file_list):
        selected = " selected" if i == 0 else ""
        dropdown_html += f'<option value="{filename}"{selected}>{model_list[i]} {vars_list[i]}</option>'
    return dropdown_html


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
    return sorted(plt.colormaps())


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
        figsize = (image_width, image_height)
        dpi = 100

        plot_grid_ref = items_dict["plot_grid_ref"]
        units = items_dict["units"]
        title = items_dict["title"]

        unit_standard, depth_factor = unit_conversion_factor(
            plot_data["depth"].attrs["units"], units
        )
        logger.debug(
            f"depth units: {plot_data['depth'].attrs['units']} to {unit_standard}, {depth_factor}"
        )
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
            if 'standard_name' in new_depth.attrs:
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
            plot_data.depth.attrs["units"] = unit_standard

            # Update to the new long name.
            if 'standard_name' in plot_data.depth.attrs:
                plot_data.depth.attrs["long_name"] = (
                    f"{plot_data.depth.attrs['standard_name']} ({unit_standard})"
                )
            else:
                plot_data.depth.attrs["long_name"] = (
                    f"{plot_data.depth.attrs['variable']} ({unit_standard})"
                )

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
        logger.debug(f"\n\n{plot_data}\nAttrs: {plot_data.attrs}")
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
    logger.debug(f"\n\nMETA: {meta}")
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
        depth_closest = closest(list(plot_data["depth"].data), depth)
        plot_data = plot_data.where(plot_data.depth == depth_closest, drop=True)
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
                            "orientation": "horizontal",
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
                    )
                else:
                    plot_data.plot(
                        ax=ax,
                        cmap=cmap,
                        x=x2,
                        y=y2,
                        add_colorbar=True,
                        transform=ccrs.PlateCarree(),
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
        logger.warning(f"figsize = {fig_width}, {fig_height}")

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
            logger.info(f"vmin, vmax (2): ", vmin, vmax)
            levels = np.linspace(vmin, vmax, 15)

            logger.warning(f"PLOT: plot_grid_ref: {plot_grid_ref}")
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
                # logger.warning(f"\n\n[INFO] lon_list:{lon_list}\nlat_list:{lat_list}")

            else:
                logger.warning(f"y_list: {y_list}")
                contourf = plt.contourf(
                    x_list, y_list, interpolated_values_2d, levels=levels, cmap=cmap
                )  # , x=grid_ref_dict[grid_ref]["x"], y=grid_ref_dict[grid_ref]["y"])
                logger.warning(f"x_list:{x_list}\ny_list:{y_list}")
            logger.warning(f"Plot Limits = {plt.xlim()}, {plt.ylim()}")
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
    # logger.warning(f"\n\n[INFO] colorbar_label: {colorbar_label}")
    # logger.warning(f"\n\n[INFO] colorbar_axes: {dir(colorbar_axes)}")
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
    plt.savefig(plot_buf, format="png")
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
