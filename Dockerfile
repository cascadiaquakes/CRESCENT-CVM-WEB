FROM python:3.13-slim

# Avoid .pyc and enable unbuffered logs; use headless matplotlib
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# System deps for Pillow/Matplotlib/NumPy and GEOS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev zlib1g-dev \
    libpng-dev libfreetype6-dev \
    libgeos-dev \
    libhdf5-dev libnetcdf-dev \
    libproj-dev proj-data proj-bin \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip first
RUN python -m pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY ./app /app

# Expose the port the app runs on
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
