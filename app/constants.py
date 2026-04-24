# The Lambda funtion names.
LAMBDA_FUNCTIONS = {
    "slice": {
        "dev": "cvm-data-extractor-latest-extract-slice",
        "crescent": "cvm-data-extractor-extract-slice",
    },
    "xsection": {
        "dev": "cvm-data-extractor-latest-extract-xsection",
        "crescent": "cvm-data-extractor-extract-xsection",
    },
    "volume": {
        "dev": "cvm-data-extractor-latest-extract-volume",
        "crescent": "cvm-data-extractor-extract-volume",
    },
}

# The S3 bucket names.
BUCKET_NAME = {
    "dev": "cvm-s3-data-latest-dev-us-east-2-aer1lu3eichu",
    "crescent": "cvm-s3-data-crescent-us-east-2-aer1lu3eichu",
}

PREFIX = {
    "netcdf": "data/netcdf/",
    "json": "data/json/",
    "config": "data/config",
    "geojson": "data/geojson/",
}

LOGO_FILE = "static/images/Crescent_Logos_horizontal_transparent.png"

# Logo placement in inches below the lower left corner of the plot area
LOGO_INCH_OFFSET_BELOW = 0.5

# Chunks are defined as a string. It repres]zents, z,y,x. A negative
# chuck will exclude that dimension from chunking. -1,-1,-1 -> {} as auto chunking.
CHUNKS = {"profile": "-1,256,256", "slice": "1,-1,-1", "xsection": "-1,-1,-1"}

VERSION = "v2025.167"
