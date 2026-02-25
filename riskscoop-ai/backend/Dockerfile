FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for geopandas/rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output directories
RUN mkdir -p output/geojson output/flood output/flood_stats tmp

EXPOSE 7777

CMD ["uvicorn", "my_os:app", "--host", "0.0.0.0", "--port", "7777"]
