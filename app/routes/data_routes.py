import os
import sys
import json
import traceback

from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    FileResponse,
    StreamingResponse,
)

from fastapi import HTTPException, Response
from fastapi.responses import HTMLResponse, PlainTextResponse

import io
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import xarray as xr
from pyproj import Proj, Geod

import time

# from metpy.interpolate import cross_section
import base64
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import matplotlib.pyplot as plt
from datetime import datetime, timezone

import gc

import h5py

import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import urllib.parse

import numpy as np

import boto3

s3_client = boto3.client("s3")
lambda_client = boto3.client("lambda")
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

# Print all environment variables
# Create a single string with all environment variables
env_string = "\n".join(f"[ENV] {key}: {value}" for key, value in os.environ.items())

# Get the ENVIRONMENT variable from the environment or default to 'dev'
logger.info(f"[INFO] ENVIRONMENT: {env_string}")
ACTIVE_ENVIRONMENT = os.getenv("ENVIRONMENT", "dev?")
logger.info(f"[INFO] ACTIVE_ENVIRONMENT: {ACTIVE_ENVIRONMENT}")

# The Lambda funtion names.
LAMBDA_FUNCTIONS = {"slice": {"dev": "cvm-data-extractor-initial-extract-slice", "stage": "cvm-data-extractor-extract-slice", "crescent": "cvm-data-extractor-extract-slice"},
"xsection":{"dev": "cvm-data-extractor-initial-extract-xsection", "stage": "cvm-data-extractor-extract-xsection", "crescent": "cvm-data-extractor-extract-xsection"},
"volume": {"dev": "cvm-data-extractor-initial-extract-volume", "stage": "cvm-data-extractor-extract-volume", "crescent": "cvm-data-extractor-extract-volume"}}

# The S3 bucket names.
BUCKET_NAME = {"dev":"cvm-s3-data-lifecycle-dev-us-east-2-aer1lu3eichu", 
"stage": "cvm-s3-data-stage-us-east-2-aer1lu3eichu", 
"crescent": "cvm-s3-data-crescent-us-east-2-aer1lu3eichu"}

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


def generate_unique_date_string():
    """
    Generates a unique string based on the current UTC date and time in ISO format with microseconds.
    This string can be used as part of an output file name.

    Args:
    -

    Returns:
    - str: A unique string that includes the base name and the current UTC date and time in ISO format.
    """
    current_time = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    # Replace colons (:) and remove hyphens (-) in the time portion
    unique_string = (
        current_time.replace("+00:00", "")
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
    )
    return unique_string


def unit_conversion_factor(unit_in, unit_out):
    """Check input and output unit and return the conversion factor."""

    unit = standard_units(unit_in.strip().lower())
    logger.debug(f"[DEBUG] convert units {unit} to {unit_out}")
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


def get_file_base_name(file_path):
    """
    Returns the base name of the file without the extension.

    Args:
    - file_path (str): The path to the file.

    Returns:
    - str: The base name of the file without the extension.
    """
    base_name = os.path.basename(file_path)  # Get the file name with extension
    name_without_extension = os.path.splitext(base_name)[0]  # Remove the extension
    return name_without_extension


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


def get_geocsv_metadata_from_ds(ds):
    """Compose GeoCSV style metadata for a given Dataset.

    Keyword arguments:
    ds -- [required] the xarray dataset
    """
    geocsv_metadata = list()
    for row in ds.attrs:
        # For some reason history lines are split (investigate).
        if row == "history":
            ds.attrs[row] = ds.attrs[row].replace("\n", ";")
        geocsv_metadata.append(f"# global_{row}: {ds.attrs[row]}")
    for var in ds.variables:
        if "variable" not in ds[var].attrs:
            geocsv_metadata.append(f"# {var}_variable: {var}")
            geocsv_metadata.append(f"# {var}_dimensions: {len(ds[var].dims)}")

        for att in ds[var].attrs:
            geocsv_metadata.append(f"# {var}_{att}: {ds[var].attrs[att]}")
            if att == "missing_value":
                geocsv_metadata.append(f"# {var}_missing_value: nan")
            if att == "variable":
                geocsv_metadata.append(f"# {var}_dimensions: {len(ds[var].dims)}")
                geocsv_metadata.append(f"# {var}_column: {var}")
    metadata = "\n".join(geocsv_metadata)
    metadata = f"{metadata}\n"
    return metadata


def calculate_dataset_size(data_to_return):
    total_size = 0

    for var_name, var in data_to_return.data_vars.items():
        # Calculate the number of elements
        num_elements = np.prod(var.shape)

        # Determine the size of each element (in bytes)
        dtype_size = var.dtype.itemsize

        # Calculate the total size for this variable
        var_size = num_elements * dtype_size

        # Add to the total size
        total_size += var_size

    # Convert the total size to MB
    total_size_mb = total_size / (1024**2)

    return total_size_mb


def save_data(data_to_return, output_format, output_file_prefix):
    delimiter = ","
    dataset_max_size = 400
    try:
        unique_id = generate_unique_date_string()

        if output_format == "csv":
            dataset_size = calculate_dataset_size(data_to_return)
            logger.debug(f"[DEBUG] Expected Dataset size: {dataset_size:.2f} MB")
            # Convert the data to a DataFrame and round to 3 decimal places
            df = data_to_return.to_dataframe().reset_index().round(3)
            csv_path = f"/tmp/{output_file_prefix}{unique_id}.csv"
            df.to_csv(csv_path, sep=delimiter, index=False)
            return FileResponse(
                path=csv_path,
                filename=f"{output_file_prefix}{unique_id}.csv",
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.csv"
                },
            )
        elif output_format == "netcdf":
            netcdf_path = f"/tmp/{output_file_prefix}{unique_id}.nc"
            data_to_return.to_netcdf(netcdf_path)
            return FileResponse(
                path=netcdf_path,
                filename=f"{output_file_prefix}{unique_id}.nc",
                media_type="application/netcdf",
                headers={
                    "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.nc"
                },
            )
        elif output_format == "geocsv":
            dataset_size = calculate_dataset_size(data_to_return)
            logger.debug(f"[DEBUG] Expected Dataset size: {dataset_size:.2f} MB")
            # Convert the data to a DataFrame and round to 3 decimal places
            df = data_to_return.to_dataframe().reset_index().round(3)
            metadata = get_geocsv_metadata_from_ds(data_to_return)
            geocsv_path = f"/tmp/{output_file_prefix}{unique_id}.geocsv"
            with open(geocsv_path, "w") as fp:
                init = f"# dataset: GeoCSV 2.0\n# delimiter: {delimiter}"
                fp.write(f"{init}\n{metadata}")
            df.to_csv(geocsv_path, sep=delimiter, index=False, mode="a")
            return FileResponse(
                path=geocsv_path,
                filename=f"{output_file_prefix}{unique_id}.geocsv",
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.geocsv"
                },
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


def save_data_large(data_to_return, output_format, output_file_prefix):
    delimiter = ","
    dataset_max_size = 400  # Maximum allowed size in MB
    try:
        unique_id = generate_unique_date_string()

        if output_format == "csv" or output_format == "geocsv":
            # Calculate the dataset size
            dataset_size = calculate_dataset_size(data_to_return)
            logger.debug(f"[DEBUG] Expected Dataset size: {dataset_size:.2f} MB")

            # Check if the dataset size exceeds the maximum allowed size
            if dataset_size > dataset_max_size:
                logger.error(
                    f"[ERR] Request too large: ~{4 * dataset_size:.2f} MB exceeds the limit of {dataset_max_size} MB. For large requests, netCDF or HDF5 format is recommended."
                )
                return {
                    "error": f"Request too large: ~{4 * dataset_size:.2f} MB exceeds the limit of {dataset_max_size} MB. For large requests, netCDF or HDF5 format is recommended."
                }

            # Continue with data processing
            df = data_to_return.to_dataframe().reset_index().round(3)
            if output_format == "csv":
                csv_path = f"/tmp/{output_file_prefix}{unique_id}.csv"
                df.to_csv(csv_path, sep=delimiter, index=False)
                return FileResponse(
                    path=csv_path,
                    filename=f"{output_file_prefix}{unique_id}.csv",
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.csv"
                    },
                )
            elif output_format == "geocsv":
                metadata = get_geocsv_metadata_from_ds(data_to_return)
                geocsv_path = f"/tmp/{output_file_prefix}{unique_id}.geocsv"
                with open(geocsv_path, "w") as fp:
                    init = f"# dataset: GeoCSV 2.0\n# delimiter: {delimiter}"
                    fp.write(f"{init}\n{metadata}")
                df.to_csv(geocsv_path, sep=delimiter, index=False, mode="a")
                return FileResponse(
                    path=geocsv_path,
                    filename=f"{output_file_prefix}{unique_id}.geocsv",
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.geocsv"
                    },
                )

        elif output_format == "netcdf":
            netcdf_path = f"/tmp/{output_file_prefix}{unique_id}.nc"
            data_to_return.to_netcdf(netcdf_path)
            return FileResponse(
                path=netcdf_path,
                filename=f"{output_file_prefix}{unique_id}.nc",
                media_type="application/netcdf",
                headers={
                    "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.nc"
                },
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


def save_data_large_all(data_to_return, output_format, output_file_prefix):
    delimiter = ","
    dataset_max_size_csv = 400  # Maximum allowed size in MB
    dataset_max_size_netcdf = 800  # Maximum allowed size in MB
    try:
        unique_id = generate_unique_date_string()
        if True:
            size_help = """<p>You may reduce the request size by following these steps:</p>
                            <ul>
                                <li>Reduce the requested model dimensions</li>
                                <li>Limit the number of variables</li>
                                <li>Use km.kg.sec units instead of m.g.sec</li>
                            </ul>"""
            # Calculate the dataset size
            dataset_size = calculate_dataset_size(data_to_return)
            logger.debug(f"[DEBUG] Expected Dataset size: {dataset_max_size_csv:.2f} MB")
            logger.debug(f"[DEBUG] Calculated Dataset size: {dataset_size:.2f} MB")

            # Check if the dataset size exceeds the maximum allowed size
            if (
                output_format == "csv" or output_format == "geocsv"
            ) and dataset_size > dataset_max_size_csv:
                logger.error(
                    f"[ERR] Request too large: ~{4 * dataset_size:.2f} MB exceeds the limit of {dataset_max_size_csv} MB. For large requests, use the netCDF or HDF5 format or download the full model directly."
                )
                # Return an HTML response with an error message
                return HTMLResponse(
                    content=f"""
                    <html>
                    <body>
                        <h3>Request Too Large</h3>
                        <p>Your request is approximately {4 * dataset_size:.2f} MB, which exceeds the maximum allowed size of {dataset_max_size_csv} MB.</p>
                        <p>For large requests, use either netCDF or HDF5 format or download the full model directly.</p>
                        {size_help}
                    </body>
                    </html>
                    """,
                    status_code=413,  # 413 Payload Too Large
                    media_type="text/html",  # Explicitly setting the Content-Type to text/html
                )
            elif output_format == "netcdfd" and dataset_size > dataset_max_size_netcdf:
                logger.error(
                    f"[ERR] Request too large: ~{4 * dataset_size:.2f} MB exceeds the limit of {dataset_max_size_netcdf} MB. For very large requests, use the HDF5 format or download the full model directly."
                )
                # Return an HTML response with an error message
                return HTMLResponse(
                    content=f"""
                    <html>
                    <body>
                        <h3>Request Too Large</h3>
                        <p>Your request is approximately {4 * dataset_size:.2f} MB, which exceeds the maximum allowed size of {dataset_max_size_csv} MB.</p>
                        <p>For very large requests, use the HDF5 format or download the full model directly.</p>
                        {size_help}
                    </body>
                    </html>
                    """,
                    status_code=413,  # 413 Payload Too Large
                    media_type="text/html",  # Explicitly setting the Content-Type to text/html
                )
        if output_format == "csv" or output_format == "geocsv":
            # Continue with data processing for CSV or GeoCSV
            df = data_to_return.to_dataframe().reset_index().round(3)
            if output_format == "csv":
                csv_path = f"/tmp/{output_file_prefix}{unique_id}.csv"
                df.to_csv(csv_path, sep=delimiter, index=False)
                return FileResponse(
                    path=csv_path,
                    filename=f"{output_file_prefix}{unique_id}.csv",
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.csv"
                    },
                )
            elif output_format == "geocsv":
                metadata = get_geocsv_metadata_from_ds(data_to_return)
                geocsv_path = f"/tmp/{output_file_prefix}{unique_id}.geocsv"
                with open(geocsv_path, "w") as fp:
                    init = f"# dataset: GeoCSV 2.0\n# delimiter: {delimiter}"
                    fp.write(f"{init}\n{metadata}")
                df.to_csv(geocsv_path, sep=delimiter, index=False, mode="a")
                return FileResponse(
                    path=geocsv_path,
                    filename=f"{output_file_prefix}{unique_id}.geocsv",
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.geocsv"
                    },
                )

        elif output_format == "netcdf":
            netcdf_path = f"/tmp/{output_file_prefix}{unique_id}.nc"

            logger.debug(f"[DEBUG] Define the compression settings for each variable")
            # Define the compression settings for each variable
            encoding = {
                var: {"zlib": True, "complevel": 2} for var in data_to_return.data_vars
            }

            logger.debug(f"[DEBUG] Save the NetCDF file with compression")
            # Save the NetCDF file with compression
            data_to_return.to_netcdf(netcdf_path, encoding=encoding)

            # Ensure file is closed and available
            time.sleep(
                1
            )  # Optional: Add a short delay to ensure the file is fully written
            logger.debug(
                f"[DEBUG] Define a generator function to stream the file in chunks"
            )

            # Define a generator function to stream the file in chunks
            # reducing the chunk size for streaming NetCDF files to minimize memory usage.
            def iterfile():
                with open(netcdf_path, "rb") as file_like:
                    while chunk := file_like.read(1024 * 1024):  # Stream in 1 MB chunks
                        yield chunk

            return StreamingResponse(
                iterfile(),
                media_type="application/netcdf",
                headers={
                    "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.nc"
                },
            )

        elif output_format == "hdf5":
            hdf5_path = f"/tmp/{output_file_prefix}{unique_id}.h5"

            # Create the HDF5 file with compression and chunking
            with h5py.File(hdf5_path, "w") as hdf:
                logger.debug(f"[DEBUG] Adding the global attributes")
                for attr_name in data_to_return.attrs:
                    hdf.attrs[attr_name] = data_to_return.attrs[attr_name]
                logger.debug(f"[DEBUG] Added the global attributes")

                # Determine group based on data dimensions
                group_name = "MODEL" if len(data_to_return.dims) == 3 else "SURFACES"
                group = hdf.create_group(group_name)
                logger.debug(f"[DEBUG] Added the {group_name} group")

                # Save data variables with compression and chunking
                for var_name, var_data in data_to_return.items():
                    logger.debug(f"[DEBUG] Adding attributes for {var_name}")
                    dset = group.create_dataset(
                        var_name,
                        data=var_data.values,
                        compression="gzip",  # Apply gzip compression
                        compression_opts=9,  # Set compression level (1-9)
                        chunks=True,  # Enable chunking, can specify chunk size if needed
                    )
                    # Copy variable attributes
                    for attr_name in var_data.attrs:
                        dset.attrs[attr_name] = var_data.attrs[attr_name]
                    logger.debug(f"[DEBUG] Added dataset and attributes for {var_name}")

                # Save coordinate variables with compression and chunking
                for coord_name in data_to_return.coords:
                    coord_data = data_to_return.coords[coord_name]
                    dset = group.create_dataset(
                        coord_name,
                        data=coord_data.values,
                        compression="gzip",  # Apply gzip compression
                        compression_opts=9,  # Set compression level (1-9)
                        chunks=True,  # Enable chunking, can specify chunk size if needed
                    )
                    logger.debug(
                        f"[DEBUG] Added dataset and attributes for {coord_name}"
                    )

            # Ensure file is closed and available
            time.sleep(1)  # Add a short delay to ensure the file is flushed

            logger.debug(f"[DEBUG] returning the response")

            # Define a generator function to stream the file in chunks
            def iterfile():
                try:
                    with open(hdf5_path, "rb") as file_like:
                        while chunk := file_like.read(
                            1024 * 1024
                        ):  # Stream in 1 MB chunks
                            yield chunk
                except Exception as e:
                    logger.error(f"[ERR] Error streaming file: {str(e)}")
                    raise

            return StreamingResponse(
                iterfile(),
                media_type="application/x-hdf5",
                headers={
                    "Content-Disposition": f"attachment; filename={output_file_prefix}{unique_id}.h5"
                },
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


def subsetter(sliced_data, limits):
    """
    Subset a dataset as a volume.

    Call arguments:
        ds - [required] the xarray dataset
        limits - [required] limits of the volume in all directions.
    """
    geospatial_dict = {
        "latitude": ["geospatial_lat_min", "geospatial_lat_max"],
        "longitude": ["geospatial_lon_min", "geospatial_lon_max"],
        "depth": ["geospatial_vertical_min", "geospatial_vertical_max"],
    }
    # Check if the array has any zero-sized dimensions
    warnings = ""

    try:
        limit_keys = list(limits.keys())
        limit_values = list(limits.values())

        logger.debug(
            f"""[DEBUG] EXTRACTING BASED ON
            ({limit_keys[0]} >= {limit_values[0][0]})
            & ({limit_keys[0]} <= {limit_values[0][1]})
            drop=True, from {sliced_data[limit_keys[0]].data}\n\n\n"""
        )

        # Apply the first filter
        sliced_data = sliced_data.where(
            (sliced_data[limit_keys[0]] >= limit_values[0][0])
            & (sliced_data[limit_keys[0]] <= limit_values[0][1]),
            drop=True,
        )

        logger.debug(
            f"""[DEBUG] EXTRACTING BASED ON
            ({limit_keys[1]} >= {limit_values[1][0]})
            & ({limit_keys[1]} <= {limit_values[1][1]})
            drop=True, from {sliced_data[limit_keys[1]].data}\n\n\n"""
        )
        # Apply the second filter
        sliced_data = sliced_data.where(
            (sliced_data[limit_keys[1]] >= limit_values[1][0])
            & (sliced_data[limit_keys[1]] <= limit_values[1][1]),
            drop=True,
        )

        logger.debug(
            f"""[DEBUG] EXTRACTING BASED ON
            ({limit_keys[2]} >= {limit_values[2][0]})
            & ({limit_keys[2]} <= {limit_values[2][1]})
            drop=True, from {sliced_data[limit_keys[2]].data}\n\n\n"""
        )
        # Apply the third filter
        sliced_data = sliced_data.where(
            (sliced_data[limit_keys[2]] >= limit_values[2][0])
            & (sliced_data[limit_keys[2]] <= limit_values[2][1]),
            drop=True,
        )

        for dim in limit_keys:
            if dim in geospatial_dict:
                #  The dropna method is used to remove coordinates with all NaN values along the specified dimensions
                sliced_data = sliced_data.dropna(dim=dim, how="all")
                if geospatial_dict[dim][0] in sliced_data.attrs:
                    sliced_data.attrs[geospatial_dict[dim][0]] = min(
                        sliced_data[dim].values
                    )
                if geospatial_dict[dim][1] in sliced_data.attrs:
                    sliced_data.attrs[geospatial_dict[dim][1]] = max(
                        sliced_data[dim].values
                    )
    except Exception as ex:
        warnings = ex
        return sliced_data, warnings

    return sliced_data, warnings


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
        interpolated_ds = ds.interp(x=path.x, y=path.y, method=method)

    return interpolated_ds, lat_points, lon_points


@router.get("/volume-data", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("volume-data.html", {"request": request})


@router.get("/3d", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("repo-3d.html", {"request": request})


@router.get("/depth-slice-data", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("depth-slice-data.html", {"request": request})


@router.get("/x-section-data", response_class=HTMLResponse)
async def read_page1(request: Request):
    return templates.TemplateResponse("x-section-data.html", {"request": request})


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


@router.get("/download-netcdf/")
def download_netcdf(filename: str):
    # Define the directory where the NetCDF files are stored
    netcdf_directory = NETCDF_DIR
    file_name = os.path.splitext(os.path.basename(filename))[0]
    # Construct the full file path
    file_path = os.path.join(netcdf_directory, f"{file_name}.nc")
    logger.debug(f"[DEBUG] netCDF file {file_path}")

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Return the file as a FileResponse
    return FileResponse(
        path=file_path,
        filename=f"{file_name}.nc",
        media_type="application/netcdf",
        headers={"Content-Disposition": f"attachment; filename={file_name}.nc"},
    )


@router.get("/extract-volume-data")
async def volume_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    end_depth: float,
    units: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
    interpolation_method: str = Query("none"),
):
    version = "v.2024.102"
    interpolation_method == "none"

    try:
        variables_to_keep = variables_hidden.split(",")
        data_file = urllib.parse.unquote(data_file)  # Decode the URL-safe string

        ds = xr.load_dataset(os.path.join("static", "netcdf", data_file))

        # We will be working with the variable dataset, so capture the metadata for the main dataset now.
        meta = ds.attrs

        # Convert start_depth units to the desired standard. start_depth is always in KM.
        _, start_depth_factor = unit_conversion_factor("km", units)

        # Data depth will be updated, we change the start depth too.
        start_depth *= start_depth_factor
        end_depth *= start_depth_factor

        # Convert depth units to the desired standard
        unit_standard, depth_factor = unit_conversion_factor(
            ds["depth"].attrs["units"], units
        )
        logger.debug(
            f"[DEBUG] DEPTH: Convert depth units to the desired standard for output. From: {ds['depth']} {ds['depth'].attrs['units']} To: {units}, unit_standard: {unit_standard}, depth_factor: {depth_factor}. start_depth: {start_depth}, start_depth_factor: {start_depth_factor}"
        )

        if depth_factor != 1:
            # Apply depth conversion factor if needed
            new_depth_values = ds["depth"] * depth_factor
            new_depth = xr.DataArray(
                new_depth_values, dims=["depth"], attrs=ds["depth"].attrs
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
            # Assign new depth coordinates with converted units
            ds = ds.assign_coords(depth=new_depth)
        else:
            # If no conversion is needed, just update units
            ds.depth.attrs["units"] = unit_standard
            if "standard_name" in ds.depth.attrs:
                ds.depth.attrs["long_name"] = (
                    f"{ds.depth.attrs['standard_name']} ({unit_standard})"
                )
            else:
                ds.depth.attrs["long_name"] = (
                    f"{new_depth.attrs['variable']} [{unit_standard}]"
                )
        # Extract grid ref metadata
        grid_ref= meta.get("grid_ref", "latitude_longitude")
        logger.debug(f"[DEBUG] Meta: {meta}\n\ngrid_ref:{grid_ref}")
        # If data is not "latitude_longitude", we need to get the start and end in the primary axis.
        if grid_ref != "latitude_longitude":
            utm_zone = meta.get("utm_zone")
            ellipsoid = meta.get("ellipsoid")
            start_x, start_y = project_lonlat_utm(
                start_lng, start_lat, utm_zone, ellipsoid
            )
            end_x, end_y = project_lonlat_utm(end_lng, end_lat, utm_zone, ellipsoid)
            x = "x"
            y = "y"
        else:
            start_x = start_lng
            start_y = start_lat
            end_x = end_lng
            end_y = end_lat
            x = "longitude"
            y = "latitude"

        subset_limits = dict()
        subset_limits[x] = [start_x, end_x]
        subset_limits[y] = [start_y, end_y]
        subset_limits["depth"] = [start_depth, end_depth]
        for var in variables_to_keep:
            # Convert the variable unit to the desired standard
            _, var_factor = unit_conversion_factor(ds[var].attrs["units"], units)
            logger.debug(
                f"[DEBUG] Convert {var} units to the desired standard for output. From: {ds[var].attrs['units']} To: {units}, unit_standard: {_}, depth_factor: {var_factor}. "
            )
            ds[var].data *= var_factor

        # Initially estimate the output dimension sizes.
        limits = {
            "csv": 1000000,
            "geocsv": 1000000,
            "netcdf": 350000000,
            "hdf5": 350000000,
        }
        sliced_data = ds[variables_to_keep]

        limit_keys = list(subset_limits.keys())
        limit_values = list(subset_limits.values())
        count_after_first_filter = (
            (
                (sliced_data[limit_keys[0]] >= limit_values[0][0])
                & (sliced_data[limit_keys[0]] <= limit_values[0][1])
            )
            .sum()
            .item()
        )
        logger.debug(
            f"Estimated output count_after_first_filter: {count_after_first_filter}"
        )
        count_after_second_filter = (
            (
                (sliced_data[limit_keys[1]] >= limit_values[1][0])
                & (sliced_data[limit_keys[1]] <= limit_values[1][1])
            )
            .sum()
            .item()
        )

        count_after_third_filter = (
            (
                (sliced_data[limit_keys[2]] >= limit_values[2][0])
                & (sliced_data[limit_keys[2]] <= limit_values[2][1])
            )
            .sum()
            .item()
        )

        count_after_all_filters = (
            count_after_first_filter
            * count_after_second_filter
            * count_after_third_filter
            * len(variables_to_keep)
        )

        logger.debug(f"Estimated output grid dimension: {count_after_all_filters}")

        # Check if the dataset size exceeds the maximum allowed size
        if True:
            size_help = """<p>You may reduce the request size by following these steps:</p>
                            <ul>
                                <li>Reduce the requested model dimensions</li>
                                <li>Limit the number of variables</li>
                                <li>Use km.kg.sec units instead of m.g.sec</li>
                                <li>For large requests use netCDF or HDF5 format</li>
                            </ul>"""
            if count_after_all_filters > limits[output_format]:
                logger.error(
                    f"[ERR] Request too large: Estimated output data points ~{count_after_all_filters:,}  exceeds the limit of {limits[output_format]:,} for {output_format} format. For large requests, use the netCDF or HDF5 format or download the full model directly."
                )
                # Return an HTML response with an error message
                return HTMLResponse(
                    content=f"""
                    <html>
                    <body>
                        <h3>Request Too Large</h3>
                        <p>Your request has approximately {count_after_all_filters:,} data points, which exceeds the maximum allowed size of {limits[output_format]:,} for {output_format} format.</p>
                        {size_help}
                    </body>
                    </html>
                    """,
                    status_code=413,  # 413 Payload Too Large
                    media_type="text/html",  # Explicitly setting the Content-Type to text/html
                )

        output_data, warnings = subsetter(ds[variables_to_keep], subset_limits)
        # output_data = subset_volume[variables_to_keep]
        meta = output_data.attrs
        logger.debug(f"[DEBUG] output_data: {output_data}")
        unique_id = generate_unique_date_string()
    except Exception as ex:
        logger.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(f"[ERR] {ex}\n{traceback.print_exc()}"),
            media_type="image/png",
        )
    output_file_prefix = f"{get_file_base_name(data_file)}_volume_data_"
    logger.debug(
        f"[DEBUG] Extraction complete, output_file_prefix:{output_file_prefix}"
    )
    logger.debug(f"[DEBUG] Use save_data_large_all")
    # Delete the DataSet to free up memory
    del ds
    gc.collect()  # Force garbage collection

    return save_data_large_all(output_data, output_format, output_file_prefix)

@router.get("/extract-volume-data-s3")
async def volume_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    end_depth: float,
    units: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query("csv"),  # Default output format is CSV
    interpolation_method: str = Query("none"),
):
    version = "v.2024.102"
    bucket_name = BUCKET_NAME[ACTIVE_ENVIRONMENT]

    # Prepare Lambda event payload
    event = {
        "bucket_name": bucket_name,
        "data_file": data_file,
        "start_lat": start_lat,
        "start_lng": start_lng,
        "end_lat": end_lat,
        "end_lng": end_lng,
        "start_depth": start_depth,
        "end_depth": end_depth,
        "variables_hidden": variables_hidden,
        "interpolation_method": interpolation_method,
        "output_format": output_format,
        "units": units,
    }

    logger.info(f"[INFO] Invoking Lambda with event: {event}")

    try:
        # Invoke Lambda function
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTIONS["volume"][ACTIVE_ENVIRONMENT],
            InvocationType="RequestResponse",
            Payload=json.dumps(event),
        )

        # Read and decode Lambda response
        lambda_response = json.loads(response['Payload'].read().decode('utf-8'))

        # Extract status code and response body
        status_code = lambda_response.get("statusCode", 500)  # Default to 500 if missing

        try:
            body_content = lambda_response.get("body", "{}")

            # Ensure body_content is a string before parsing
            if isinstance(body_content, dict):
                parsed_body = body_content  # Already a valid JSON object
            elif isinstance(body_content, str) and body_content.startswith("{"):
                parsed_body = json.loads(body_content)  # Properly formatted JSON string
            else:
                parsed_body = {"error": body_content}  # Handle plain text errors

        except json.JSONDecodeError:
            logger.error("[ERROR] Invalid JSON format in Lambda response body")
            raise HTTPException(status_code=500, detail="Lambda returned invalid JSON")

        # Log Lambda response without exposing full details
        logger.info(f"[INFO] Lambda response status, status_code: {status_code}")
        logger.info(f"[INFO] Lambda response status, parsed_body: {parsed_body}")


        # Raise HTTPException with properly formatted JSON response
        if status_code != 200:
            logger.error(f"[ERROR] Lambda returned error {status_code}: {parsed_body}")
            return JSONResponse(status_code=status_code, content=parsed_body)

        # Extract S3 key
        s3_key = parsed_body.get("s3_key")
        if not s3_key:
            logger.error("[ERROR] Missing 's3_key' in Lambda response")
            raise HTTPException(status_code=500, detail="Lambda response missing 's3_key'")

        # Fetch the file from S3
        try:
            s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            file_stream = io.BytesIO(s3_response['Body'].read())
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"[ERROR] S3 key {s3_key} not found in {bucket_name}")
            raise HTTPException(status_code=404, detail=f"S3 key {s3_key} not found in {bucket_name}")

        # Delete the file from S3 after retrieval
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logger.warning(f"[WARN] Removed the output file {s3_key} from {bucket_name} bucket")
        except s3_client.exceptions.NoSuchKey:
            logger.warning(f"[WARN] Attempted to delete non-existent S3 key: {s3_key}")

        # Determine media type based on output format
        media_type = 'text/csv' if output_format == 'csv' else 'application/netcdf'

        # Serve the file to the user
        return StreamingResponse(
            file_stream,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={s3_key.split('/')[-1]}"
            }
        )

    except Exception as e:
        error_details = traceback.format_exc()  # Get full traceback
        logger.error(f"[ERROR] Exception occurred: {error_details}")  # Log detailed error
        raise HTTPException(status_code=500, detail="Internal Server Error")

        
@router.get("/fetch_json", response_model=dict)
def fetch_json(file_name: str):
    """
    Fetches the JSON content from S3 under the 'data/json/' directory for the given file name.
    """
    bucket_name = BUCKET_NAME[ACTIVE_ENVIRONMENT]
    JSON_PREFIX = "data/json/"

    # Construct the full S3 key for the requested JSON file
    s3_key = f"{JSON_PREFIX}{file_name}"#.json"

    try:
        # Fetch JSON file from S3
        json_object = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        json_data = json.loads(json_object["Body"].read().decode("utf-8"))
        return json_data

    except NoCredentialsError:
        logger.error("AWS credentials not found. Ensure they are properly configured.")
        raise HTTPException(status_code=500, detail="AWS credentials missing")

    except s3_client.exceptions.NoSuchKey:
        logger.error(f"File {s3_key} not found in S3 bucket {bucket_name}")
        raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        logger.error(f"Error accessing S3: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving JSON file")

# Route to create html for model dropdown. A simple drop-down of the file names and model variables.
@router.get("/models_drop_down_s3", response_class=HTMLResponse)
def models_drop_down_s3(required_variable: Optional[str] = None):
    logger.debug(f"Received route_name: {required_variable}")

    # AWS S3 configurations
    bucket_name = BUCKET_NAME[ACTIVE_ENVIRONMENT]
    JSON_PREFIX = "data/json/"
    NETCDF_PREFIX = "data/netcdf/"

    file_list = []
    vars_list = []
    model_list = []

    try:
        # List JSON files in the S3 bucket
        json_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=JSON_PREFIX)
        if "Contents" not in json_objects:
            logger.warning(f"No JSON files found in S3 bucket {bucket_name}")
            return "<option value=''>No models available</option>"

        json_files = [obj["Key"] for obj in json_objects["Contents"] if obj["Key"].endswith(".json")]

        # List NetCDF files in the S3 bucket
        netcdf_objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=NETCDF_PREFIX)
        if "Contents" in netcdf_objects:
            netcdf_files = {obj["Key"] for obj in netcdf_objects["Contents"]}  # Use a set for quick lookups
        else:
            netcdf_files = set()
        
        for json_file in sorted(json_files):
            nc_filename = json_file.replace(JSON_PREFIX, "").replace(".json", ".nc")
            nc_s3_key = f"{NETCDF_PREFIX}{nc_filename}"
            logger.warning(f"Working on {json_file} and the key: {nc_s3_key} in {bucket_name}.")

            # Check if the corresponding NetCDF file exists in the pre-fetched list
            if nc_s3_key not in netcdf_files:
                logger.warning(f"NetCDF file {nc_filename} not found in S3 {bucket_name}, with key {nc_s3_key}. Skipping.")
                continue

            # Read JSON data from S3
            json_object = s3_client.get_object(Bucket=bucket_name, Key=json_file)
            json_data = json.loads(json_object["Body"].read().decode("utf-8"))

            # Skip files that do not contain the required variable
            if required_variable and required_variable not in json_data.get("vars", []):
                logger.warning(
                    f"[WARN] Model {json_file} is rejected since the required variable '{required_variable}' "
                    f"was not in '{json_data.get('vars', [])}'"
                )
                continue

            file_list.append(nc_filename)
            data_vars = f"({','.join(json_data.get('data_vars', []))})"
            model_name = json_data.get("model", "Unknown Model")
            vars_list.append(data_vars)
            model_list.append(model_name)

    except NoCredentialsError:
        logger.error("AWS credentials not found. Ensure they are properly configured.")
        return "<option value=''>Error: AWS credentials missing</option>"

    except Exception as e:
        logger.error(f"Error accessing S3: {e}")
        return "<option value=''>Error retrieving models</option>"

    # Prepare the HTML for the dropdown
    dropdown_html = ""
    for i, filename in enumerate(file_list):
        selected = " selected" if i == 0 else ""
        dropdown_html += f'<option value="{filename}"{selected}>{model_list[i]}</option>'

    logger.debug(f"dropdown_html {dropdown_html}")
    return dropdown_html


@router.get("/extract-slice-data")
async def extract_slice_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    lng_points: int,
    lat_points: int,
    units: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query("csv"),  # Default output format is CSV
    interpolation_method: str = Query("none"),  # Default interpolation method is "none"
):
    version = "v.2024.102"  # API version for tracking

    try:
        # Decode the URL-safe string for the data file path
        data_file = urllib.parse.unquote(data_file)
        # Load the dataset using xarray
        output_data = xr.load_dataset(os.path.join("static", "netcdf", data_file))
        logger.warning(f"[DEBUG] GET: {locals()}")  # Log request parameters

        # Convert start_depth units to the desired standard. start_depth is always in KM.
        _, start_depth_factor = unit_conversion_factor("km", units)

        # Convert depth units to the desired standard
        unit_standard, depth_factor = unit_conversion_factor(
            output_data["depth"].attrs["units"], units
        )
        logger.debug(
            f"[DEBUG] DEPTH: Convert depth units to the desired standard for output. From: {output_data['depth'].attrs['units']} To: {units}, unit_standard: {unit_standard}, depth_factor: {depth_factor}. start_depth: {start_depth}, start_depth_factor: {start_depth_factor}"
        )

        # Data depth will be updated, we change the start depth too.
        start_depth *= start_depth_factor

        if depth_factor != 1:
            # Apply depth conversion factor if needed
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
            # Assign new depth coordinates with converted units
            output_data = output_data.assign_coords(depth=new_depth)
        else:
            # If no conversion is needed, just update units
            output_data.depth.attrs["units"] = unit_standard
            if "standard_name" in output_data.depth.attrs:
                output_data.depth.attrs["long_name"] = (
                    f"{output_data.depth.attrs['standard_name']} ({unit_standard})"
                )
            else:
                output_data.depth.attrs["long_name"] = (
                    f"{output_data.depth.attrs['variable']} ({unit_standard})"
                )
        meta = output_data.attrs  # Extract metadata from the dataset

        # Split the hidden variables list by commas
        variable_list = variables_hidden.split(",")
        selected_data_vars = output_data[
            variable_list
        ]  # Select only the specified variables

        # Apply unit conversion to each selected variable
        for var in variable_list:
            unit_standard, var_factor = unit_conversion_factor(
                selected_data_vars[var].attrs["units"], units
            )
            logger.debug(
                f"[DEBUG] VARS: Convert {var} units to the desired standard for output. From: {output_data[var].attrs['units']} To: {units}, unit_standard: {unit_standard}, depth_factor: {depth_factor}"
            )
            # Adjust data by the conversion factor
            selected_data_vars[var].data *= var_factor
            # Update units and display name with the new units
            selected_data_vars[var].attrs["units"] = unit_standard
            selected_data_vars[var].attrs[
                "display_name"
            ] = f"{selected_data_vars[var].attrs['long_name']} [{unit_standard}]"

    except Exception as ex:
        logger.error(
            f"[ERR] {ex}"
        )  # Log any errors encountered during data preparation
        return Response(
            content=create_error_image(
                f"[ERR] Bad selection: \n{ex}\n{traceback.print_exc()}"
            ),
            media_type="image/png",
        )  # Return an error image in case of failure

    try:
        # Extract grid ref metadata
        grid_ref = meta.get("output_grid_ref", "latitude_longitude")
        utm_zone = meta.get("utm_zone")
        ellipsoid = meta.get("ellipsoid")

        # Generate coordinate grids based on grid ref type
        if grid_ref == "latitude_longitude":
            # For lat/lon grid, generate lists of latitudes and longitudes
            lon_list = np.linspace(start_lng, end_lng, lng_points).tolist()
            lat_list = np.linspace(start_lat, end_lat, lat_points).tolist()
            x_list, y_list = project_lonlat_utm(lon_list, lat_list, utm_zone, ellipsoid)
            x_grid, y_grid = np.meshgrid(
                lon_list, lat_list
            )  # Create a meshgrid for lat/lon
        else:
            # For UTM grid, project lat/lon to UTM coordinates
            start_x, start_y = project_lonlat_utm(
                start_lng, start_lat, utm_zone, ellipsoid
            )
            end_x, end_y = project_lonlat_utm(end_lng, end_lat, utm_zone, ellipsoid)
            x_list = np.linspace(start_x, end_x, lng_points).tolist()
            y_list = np.linspace(start_y, end_y, lat_points).tolist()
            # Convert UTM coordinates back to lat/lon
            lon_list, lat_list = project_lonlat_utm(
                x_list, y_list, utm_zone, ellipsoid, xy_to_latlon=True
            )
            x_grid, y_grid = np.meshgrid(x_list, y_list)  # Create a meshgrid for UTM

        interp_type = interpolation_method  # Set interpolation method

        if interp_type == "none":
            # If no interpolation, find the closest grid points for slicing
            start_lat = closest(output_data["latitude"].data, start_lat)
            end_lat = closest(output_data["latitude"].data, end_lat)
            start_lng = closest(output_data["longitude"].data, start_lng)
            end_lng = closest(output_data["longitude"].data, end_lng)

            depth_closest = closest(list(output_data["depth"].data), start_depth)
            logger.debug(
                f"[DEBUG] Depth: {output_data['depth']}\nLooking for: {start_depth}\nClosest: {depth_closest}"
            )
            # Filter data by depth and bounding box
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
            data_to_return = selected_data_vars  # Return filtered data
        else:
            # If interpolation is required
            interpolated_values_2d = {}
            for var in variable_list:
                # Interpolate based on latitude/longitude or UTM coordinates
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

            # Check for NaN values in the interpolated data
            if any(
                np.all(np.isnan(interpolated_values_2d[var])) and interp_type == "cubic"
                for var in variable_list
            ):
                message = "[ERR] Data with NaN values. Can't use the cubic spline interpolation"
                logger.error(message)  # Log the error
                # Create error image and CSV data for response
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
                gc.collect()  # Clean up resources
                return JSONResponse(
                    content=content
                )  # Return JSON response with error image and CSV
            data_to_return = xr.Dataset(
                interpolated_values_2d
            )  # Return interpolated data as dataset
    except Exception as ex:
        logger.error(f"[ERR] {ex}")  # Log any errors encountered during data processing
        return Response(
            content=create_error_image(f"[ERR] {ex}\n{traceback.print_exc()}"),
            media_type="image/png",
        )  # Return an error image in case of failure

    output_file_prefix = f"{get_file_base_name(data_file)}_depth-slice-data_"  # Generate a prefix for the output file
    return save_data_large_all(
        data_to_return, output_format, output_file_prefix
    )  # Save and return the data in the desired format

# Extract slice data via s3 data.
@router.get("/extract-slice-data-s3")
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
    units: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query("csv"),  # Default output format is CSV
    interpolation_method: str = Query("none"),  # Default interpolation method is "none"
):
    # Prepare the event payload for Lambda
    bucket_name = BUCKET_NAME[ACTIVE_ENVIRONMENT]
    event = {
        "bucket_name": bucket_name,
        "data_file": data_file,
        "start_lat": start_lat,
        "start_lng": start_lng,
        "end_lat": end_lat,
        "end_lng": end_lng,
        "start_depth": start_depth,
        "lat_points": lat_points,
        "lng_points": lng_points,
        "variables_hidden": variables_hidden,
        "interpolation_method": interpolation_method,
        "output_format": output_format,
        "units": units
    }
    logger.info(f"[INFO] Invoking Lambda with event: {event}")

    try:
        # Invoke Lambda function
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTIONS["slice"][ACTIVE_ENVIRONMENT],
            InvocationType="RequestResponse",
            Payload=json.dumps(event),
        )

        # Read and decode Lambda response
        lambda_response = json.loads(response['Payload'].read().decode('utf-8'))

        # Extract status code and response body
        status_code = lambda_response.get("statusCode", 500)  # Default to 500 if missing

        try:
            body_content = lambda_response.get("body", "{}")

            # Ensure body_content is a string before parsing
            if isinstance(body_content, dict):
                parsed_body = body_content  # Already a valid JSON object
            elif isinstance(body_content, str) and body_content.startswith("{"):
                parsed_body = json.loads(body_content)  # Properly formatted JSON string
            else:
                parsed_body = {"error": body_content}  # Handle plain text errors

        except json.JSONDecodeError:
            logger.error("[ERROR] Invalid JSON format in Lambda response body")
            raise HTTPException(status_code=500, detail="Lambda returned invalid JSON")

        # Log Lambda response without exposing full details
        logger.info(f"[INFO] Lambda response status, status_code: {status_code}")
        logger.info(f"[INFO] Lambda response status, parsed_body: {parsed_body}")


        # Raise HTTPException with properly formatted JSON response
        if status_code != 200:
            logger.error(f"[ERROR] Lambda returned error {status_code}: {parsed_body}")
            return JSONResponse(status_code=status_code, content=parsed_body)

        # Extract S3 key
        s3_key = parsed_body.get("s3_key")
        if not s3_key:
            logger.error("[ERROR] Missing 's3_key' in Lambda response")
            raise HTTPException(status_code=500, detail="Lambda response missing 's3_key'")

        # Fetch the file from S3
        try:
            s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            file_stream = io.BytesIO(s3_response['Body'].read())
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"[ERROR] S3 key {s3_key} not found in {bucket_name}")
            raise HTTPException(status_code=404, detail=f"S3 key {s3_key} not found in {bucket_name}")

        # Delete the file from S3 after retrieval
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logger.warning(f"[WARN] Removed the output file {s3_key} from {bucket_name} bucket")
        except s3_client.exceptions.NoSuchKey:
            logger.warning(f"[WARN] Attempted to delete non-existent S3 key: {s3_key}")

        # Determine media type based on output format
        media_type = 'text/csv' if output_format == 'csv' else 'application/netcdf'

        # Serve the file to the user
        return StreamingResponse(
            file_stream,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={s3_key.split('/')[-1]}"
            }
        )

    except Exception as e:
        error_details = traceback.format_exc()  # Get full traceback
        logger.error(f"[ERROR] Exception occurred: {error_details}")  # Log detailed error
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/extract-xsection-data-s3")
async def xsection_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    end_depth: float,
    num_points: int,
    units: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
    interpolation_method: str = Query("none"),
):
    version = "v.2024.102"

    # Prepare the event payload for Lambda
    bucket_name = BUCKET_NAME[ACTIVE_ENVIRONMENT]
    event = {
        "bucket_name": bucket_name,
        "data_file": data_file,
        "start_lat": start_lat,
        "start_lng": start_lng,
        "end_lat": end_lat,
        "end_lng": end_lng,
        "start_depth": start_depth,
        "end_depth": end_depth,
        "num_points": num_points,
        "variables_hidden": variables_hidden,
        "interpolation_method": interpolation_method,
        "output_format": output_format,
        "units": units
    }
    logger.info(f"[INFO] Invoking Lambda with event: {event}")

    try:
        # Invoke Lambda function
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTIONS["xsection"][ACTIVE_ENVIRONMENT],
            InvocationType="RequestResponse",
            Payload=json.dumps(event),
        )

        # Read and decode Lambda response
        lambda_response = json.loads(response['Payload'].read().decode('utf-8'))

        # Extract status code and response body
        status_code = lambda_response.get("statusCode", 500)  # Default to 500 if missing

        try:
            body_content = lambda_response.get("body", "{}")

            # Ensure body_content is a string before parsing
            if isinstance(body_content, dict):
                parsed_body = body_content  # Already a valid JSON object
            elif isinstance(body_content, str) and body_content.startswith("{"):
                parsed_body = json.loads(body_content)  # Properly formatted JSON string
            else:
                parsed_body = {"error": body_content}  # Handle plain text errors

        except json.JSONDecodeError:
            logger.error("[ERROR] Invalid JSON format in Lambda response body")
            raise HTTPException(status_code=500, detail="Lambda returned invalid JSON")

        # Log Lambda response without exposing full details
        logger.info(f"[INFO] Lambda response status, status_code: {status_code}")
        logger.info(f"[INFO] Lambda response status, parsed_body: {parsed_body}")


        # Raise HTTPException with properly formatted JSON response
        if status_code != 200:
            logger.error(f"[ERROR] Lambda returned error {status_code}: {parsed_body}")
            return JSONResponse(status_code=status_code, content=parsed_body)

        # Extract S3 key
        s3_key = parsed_body.get("s3_key")
        if not s3_key:
            logger.error("[ERROR] Missing 's3_key' in Lambda response")
            raise HTTPException(status_code=500, detail="Lambda response missing 's3_key'")

        # Fetch the file from S3
        try:
            s3_response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            file_stream = io.BytesIO(s3_response['Body'].read())
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"[ERROR] S3 key {s3_key} not found in {bucket_name}")
            raise HTTPException(status_code=404, detail=f"S3 key {s3_key} not found in {bucket_name}")

        # Delete the file from S3 after retrieval
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logger.warning(f"[WARN] Removed the output file {s3_key} from {bucket_name} bucket")
        except s3_client.exceptions.NoSuchKey:
            logger.warning(f"[WARN] Attempted to delete non-existent S3 key: {s3_key}")

        # Determine media type based on output format
        media_type = 'text/csv' if output_format == 'csv' else 'application/netcdf'

        # Serve the file to the user
        return StreamingResponse(
            file_stream,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={s3_key.split('/')[-1]}"
            }
        )

    except Exception as e:
        error_details = traceback.format_exc()  # Get full traceback
        logger.error(f"[ERROR] Exception occurred: {error_details}")  # Log detailed error
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Extract an interpolated cross-section.
@router.get("/extract-xsection-data")
async def extract_xsection_data(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    end_depth: float,
    num_points: int,
    units: str,
    interpolation_method: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
):
    # Extract the cross-section.
    version = "v.2024.102"
    version_label = f"{version}  {utc_now()['date_time']} UTC"
    form_data = await request.form()

    items_dict = dict(form_data.items())
    logger.debug(f"[DEBUG] POST: {items_dict}")

    try:
        # lazy loading with Dask.
        ds = xr.open_dataset(os.path.join("static", "netcdf", data_file), chunks={})

        start = [start_lat, start_lng]
        end = [end_lat, end_lng]

        # Requested depth range is always in km.
        _, depth_range_factor = unit_conversion_factor("km", units)
        depth = [start_depth * depth_range_factor, end_depth * depth_range_factor]

        units = units
        logger.debug(
            f"\n\n[DEBUG] converting the depth units {ds['depth']},\nunits: {ds['depth'].attrs['units']}"
        )
        unit_standard, depth_factor = unit_conversion_factor(
            ds["depth"].attrs["units"], units
        )
        logger.debug(
            f"[DEBUG] depth units: {ds['depth'].attrs['units']},  {depth_factor}"
        )
        ds["depth"].attrs["units"] = unit_standard

        ds["depth"] = ds["depth"] * depth_factor
        ds["depth"].attrs["units"] = units

        ds = ds.where(
            (ds.depth >= float(depth[0])) & (ds.depth <= float(depth[1])),
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
    meta = ds.attrs
    dpi = 100

    logger.debug(f"\n\n[DEBUG] META: {meta}")
    if "grid_ref" not in meta:
        logger.debug(
            f"[DEBUG] The 'grid_ref' attribute not found. Assuming geographic coordinate system"
        )
        grid_ref = "latitude_longitude"
    else:
        grid_ref = meta["grid_ref"]

        # Cross-section interpolation type.
        interp_type = interpolation_method

        # Steps in the cross-section.
        steps = num_points
        logger.info(
            f"[INFO] cross_section start:{start}, end: {end}, steps: {steps}, interp_type: {interp_type}, ds: {ds}"
        )
        # Extract the cross-section.
        logger.info(
            f"[INFO] Before cross_section (ds,start,end,steps,interp_type:)\n{ds},{start},{end},{steps},{interp_type}"
        )
        try:
            plot_data, latitudes, longitudes = interpolate_path(
                ds,
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

        logger.debug(
            f"\n\n[DEBUG] units: {units}, cumulative_distances: {cumulative_distances}"
        )

        # Assuming 'plot_data' is an xarray Dataset or DataArray
        # Create a new coordinate 'distance' based on the cumulative distances
        plot_data = plot_data.assign_coords(distance=("points", cumulative_distances))

        # If you want to use 'distance' as a dimension instead of 'index',
        # you can swap the dimensions (assuming 'index' is your current dimension)
        plot_data = plot_data.swap_dims({"points": "distance"})
        logger.debug(f"\n\n[DEBUG] plot_data:{plot_data}")

        # Split the hidden variables list by commas
        variable_list = variables_hidden.split(",")
        logger.debug(f"\n\n[DEBUG] variable_list:{variable_list}")
        data_to_plot = plot_data[variable_list]  # Select only the specified variables

        for var in variable_list:
            logger.info(
                f"[INFO] {var} units: {data_to_plot[var].attrs['units']}, units: {units}"
            )
            _, var_factor = unit_conversion_factor(
                data_to_plot[var].attrs["units"], units
            )
            logger.debug(
                f"\n\n[DEBUG] plot_var units: {data_to_plot[var].attrs['units']}, units: {units} => unit_standard:{unit_standard}, var_factor: {var_factor}"
            )

            # Copy all attributes from the original variable to the new one
            data_to_plot[var].attrs = ds[var].attrs.copy()
            logger.debug(f"\n\n[DEBUG] var: {var}, attr: {ds[var].attrs}")
            data_to_plot[var] = data_to_plot[var] * var_factor

            data_to_plot.attrs["units"] = unit_standard

            logger.info(f"[INFO] Cross-section input: {items_dict}")
            # logger.info(f"[INFO] Cross-section input: {items_dict['data_file']}, start: {start}, end: {end}, step: {steps}, interp_type: {interp_type}, plot_var: {plot_var}")
        logger.debug(f"\n\n[DEBUG] data_to_plot:{data_to_plot}")

        # Set the depth limits for display.
        logger.debug(f"\n\n[DEBUG] Depth limits:{depth}")

    output_file_prefix = f"{get_file_base_name(data_file)}_xsection_data_"

    return save_data_large_all(data_to_plot, output_format, output_file_prefix)


@router.get("/extract-xsection-data2")
async def extract_xsection_data2(
    request: Request,
    data_file: str,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    start_depth: float,
    end_depth: float,
    num_points: int,
    units: str,
    interpolation_method: str,
    variables_hidden: str,  # Multiple comma-separated values
    output_format: str = Query(
        "csv"
    ),  # Add output_format argument with default value as "csv"
):
    version = "v.2024.102"
    # Extract the cross-section.
    try:
        data_file = urllib.parse.unquote(data_file)  # Decode the URL-safe string
        with xr.open_dataset(os.path.join(NETCDF_DIR, data_file)) as ds:

            plot_data = ds.copy()
            meta = plot_data.attrs
            plot_data = plot_data.where(
                (plot_data.depth >= float(start_depth))
                & (plot_data.depth <= float(end_depth)),
                drop=True,
            )

            start = [start_lat, start_lng]
            end = [end_lat, end_lng]
            grid_ref = meta.get("output_grid_ref", "latitude_longitude")
            utm_zone = meta.get("utm_zone")
            ellipsoid = meta.get("ellipsoid")
            xsection_data, latitudes, longitudes = interpolate_path(
                plot_data,
                start,
                end,
                num_points=num_points,
                method=interpolation_method,
                grid_ref=grid_ref,
                utm_zone=utm_zone,
                ellipsoid=ellipsoid,
            )

            # Calculate distances between consecutive points
            geod = Geod(ellps=meta["ellipsoid"])
            _, _, distances = geod.inv(
                longitudes[:-1],
                latitudes[:-1],
                longitudes[1:],
                latitudes[1:],
            )

            # Compute cumulative distance, starting from 0
            cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

            if "grid_ref" not in meta:
                logger.warning(
                    f"[WARN] The 'grid_ref' attribute not found. Assuming geographic coordinate system"
                )
                grid_ref = "latitude_longitude"

            else:
                grid_ref = meta["grid_ref"]

            logger.warning(f"[WARN] Units: {units}")
            if units == "mks":
                cumulative_distances = cumulative_distances / 1000.0
                distance_label = "distance (km)"
            else:
                distance_label = "distance (m)"

            # Create a new coordinate 'distance' based on the cumulative distances
            xsection_data = xsection_data.assign_coords(
                distance=("points", cumulative_distances)
            )

            # If you want to use 'distance' as a dimension instead of 'index',
            # you can swap the dimensions (assuming 'index' is your current dimension)
            xsection_data = xsection_data.swap_dims({"points": "distance"})
            output_data = xsection_data.copy()
            unique_id = generate_unique_date_string()
    except Exception as ex:
        logger.error(f"[ERR] {ex}")
        return Response(
            content=create_error_image(f"[ERR] {ex}\n{traceback.print_exc()}"),
            media_type="image/png",
        )
    output_file_prefix = f"{get_file_base_name(data_file)}_xsection_data_"

    return save_data_large_all(output_data, output_format, output_file_prefix)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
