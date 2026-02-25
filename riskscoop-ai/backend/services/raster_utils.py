"""Raster utilities for processing GeoTIFF files.

This module provides utilities for masking, merging, and reprojecting raster data.
Adapted from geoforge_services for RiskScoop AI.
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.io import DatasetReader, MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import shape

logger = logging.getLogger(__name__)

# Configure GDAL for optimal COG access with HTTP caching
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '100000000'  # 100MB cache per file
os.environ['GDAL_CACHEMAX'] = '512'  # 512MB block cache
os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
os.environ['GDAL_HTTP_MULTIPLEX'] = 'YES'
os.environ['CPL_VSIL_CURL_CHUNK_SIZE'] = '1048576'  # 1MB chunks
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'


def mask_tiff_from_source(
    src: DatasetReader,
    polygon_gdf: gpd.GeoDataFrame,
    null_values: List,
    nodata_value: float,
    meta_template: dict = None,
    url: str = None,
) -> Tuple[DatasetReader, dict]:
    """
    Applies a polygon mask to an already-open rasterio source and returns an in-memory dataset.
    This is more efficient when you need to mask the same tile multiple times.

    Args:
        src: Already-open rasterio DatasetReader
        polygon_gdf: Polygon geometries to mask with
        null_values: Values to treat as null/nodata
        nodata_value: Value to use for nodata in output
        meta_template: Optional metadata template to use
        url: Optional URL for logging purposes

    Returns:
        Tuple of (in-memory dataset, metadata template)
    """
    try:
        if url:
            logger.debug(f"[raster_utils] Masking TIFF from open source: {url}")
        else:
            logger.debug("[raster_utils] Masking TIFF from open source")

        if meta_template is None:
            meta_template = src.meta.copy()

        # Handle case where raster has no CRS - assume WGS84
        if src.crs is None:
            logger.warning("[raster_utils] Raster has no CRS, assuming EPSG:4326")
            raster_crs = CRS.from_epsg(4326)
        else:
            raster_crs = src.crs

        # Reproject polygon if needed
        poly = (
            polygon_gdf
            if polygon_gdf.crs == raster_crs
            else polygon_gdf.to_crs(raster_crs)
        )
        geoms = list(poly.geometry.values)

        # Apply mask
        masked_data, masked_transform = mask(
            src, shapes=geoms, crop=True, all_touched=True, nodata=nodata_value
        )

        # Replace null values with nodata
        for value_to_replace in null_values:
            masked_data = np.where(
                masked_data == value_to_replace, nodata_value, masked_data
            )

        # Create in-memory dataset
        memfile = MemoryFile()
        masked_meta = src.meta.copy()
        update_dict = {
            "dtype": masked_data.dtype,
            "height": masked_data.shape[1],
            "width": masked_data.shape[2],
            "transform": masked_transform,
            "nodata": nodata_value,
        }
        masked_meta.update(update_dict)

        dataset = memfile.open(**masked_meta)
        dataset.write(masked_data)

        logger.debug(
            f"[raster_utils] Masked successfully. Shape: {masked_data.shape}, dtype: {dataset.dtypes[0]}"
        )
        return dataset, meta_template

    except Exception as e:
        logger.error(f"[raster_utils] Masking failed: {e}", exc_info=True)
        raise


def mask_tiff(
    url: str,
    polygon_gdf: gpd.GeoDataFrame,
    null_values: List,
    nodata_value: float,
    meta_template: dict = None,
) -> Tuple[DatasetReader, dict]:
    """
    Opens a TIFF from a URL, applies a polygon mask, and returns an in-memory dataset.
    This is a wrapper around mask_tiff_from_source that opens the URL first.

    Args:
        url: URL or path to TIFF file
        polygon_gdf: Polygon geometries to mask with
        null_values: Values to treat as null/nodata
        nodata_value: Value to use for nodata in output
        meta_template: Optional metadata template

    Returns:
        Tuple of (in-memory dataset, metadata template)
    """
    logger.debug(f"[raster_utils] Masking TIFF from URL: {url}")
    with rasterio.open(url) as src:
        return mask_tiff_from_source(
            src=src,
            polygon_gdf=polygon_gdf,
            null_values=null_values,
            nodata_value=nodata_value,
            meta_template=meta_template,
            url=url,
        )


def merge_datasets(masked_datasets: List[DatasetReader]) -> Tuple[np.ndarray, Affine]:
    """
    Merges a list of masked datasets into a mosaic.

    Args:
        masked_datasets: List of rasterio datasets to merge

    Returns:
        Tuple of (mosaic array, output transform)
    """
    if not masked_datasets:
        logger.error("[raster_utils] No masked datasets available to merge.")
        raise ValueError("No datasets to merge")

    logger.debug(f"[raster_utils] Merging {len(masked_datasets)} datasets into mosaic")
    mosaic, out_trans = merge(masked_datasets)
    logger.debug(f"[raster_utils] Merge completed. Mosaic shape: {mosaic.shape}, dtype: {mosaic.dtype}")
    return mosaic, out_trans


def reproject_mosaic(
    mosaic: np.ndarray,
    out_trans: Affine,
    meta_template: dict,
    null_values: List[float],
    nodata_value: float,
    dst_crs: str = "EPSG:4326",
    target_resolution: Optional[float] = None,
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[np.ndarray, dict, float, float]:
    """
    Reprojects the merged mosaic to the target CRS.

    Args:
        mosaic: Input raster array
        out_trans: Input transform
        meta_template: Metadata template
        null_values: Values to replace with nodata
        nodata_value: Nodata value
        dst_crs: Target CRS (default EPSG:4326 for web display)
        target_resolution: Target resolution in CRS units (None = keep native)
        resampling: Resampling method

    Returns:
        Tuple of (reprojected mosaic, metadata, min_dn, max_dn)
    """
    logger.debug(f"[raster_utils] Reprojecting mosaic to {dst_crs}")

    # Calculate min/max excluding nodata values
    valid_data = mosaic[mosaic != nodata_value]
    if valid_data.size > 0:
        min_dn = float(np.nanmin(valid_data))
        max_dn = float(np.nanmax(valid_data))
    else:
        min_dn = 0.0
        max_dn = 0.0
    logger.debug(f"[raster_utils] Mosaic DN range: {min_dn} to {max_dn}")

    # Get bounds
    bounds = rasterio.transform.array_bounds(
        mosaic.shape[1], mosaic.shape[2], out_trans
    )
    left, bottom, right, top = bounds

    # Calculate transform
    if target_resolution is None:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            meta_template["crs"],
            dst_crs,
            mosaic.shape[2],
            mosaic.shape[1],
            left,
            bottom,
            right,
            top,
        )
    else:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            meta_template["crs"],
            dst_crs,
            mosaic.shape[2],
            mosaic.shape[1],
            left,
            bottom,
            right,
            top,
            resolution=(target_resolution, target_resolution),
        )

    # Create output array
    dst_mosaic = np.empty(
        shape=(mosaic.shape[0], dst_height, dst_width), dtype=mosaic.dtype
    )

    # Reproject each band
    for i in range(mosaic.shape[0]):
        reproject(
            source=mosaic[i],
            destination=dst_mosaic[i],
            src_transform=out_trans,
            src_crs=meta_template["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    # Update metadata
    out_meta = meta_template.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "transform": dst_transform,
        "crs": dst_crs,
        "compress": "lzw",
        "nodata": nodata_value,
    })

    # Replace null values
    tolerance = 0.01
    for value in null_values:
        dst_mosaic = np.where(
            np.abs(dst_mosaic - value) < tolerance, nodata_value, dst_mosaic
        )

    logger.debug(f"[raster_utils] Reprojection completed. Output size: {dst_width}x{dst_height}")
    return dst_mosaic, out_meta, min_dn, max_dn


def save_raster_to_file(
    mosaic: np.ndarray,
    out_meta: dict,
    output_path: Path,
) -> Path:
    """
    Saves a raster array to a GeoTIFF file.

    Args:
        mosaic: Raster array to save
        out_meta: Metadata for the output file
        output_path: Path to save the file

    Returns:
        Path to the saved file
    """
    logger.debug(f"[raster_utils] Saving raster to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    logger.info(f"[raster_utils] Saved raster to {output_path}")
    return output_path


def merge_mask_each_tiff(
    tiff_urls: List[str],
    polygon_gdf: gpd.GeoDataFrame,
    output_path: Path,
    null_values: List[float],
    nodata_value: float,
    dst_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
) -> Tuple[Path, float, float, dict]:
    """
    For a list of TIFF URLs, applies the mask using the given polygon, merges the results,
    reprojects the mosaic, and saves to file.

    Args:
        tiff_urls: List of TIFF URLs to process
        polygon_gdf: Polygon to mask with
        output_path: Path for output file
        null_values: Values to treat as nodata
        nodata_value: Nodata value for output
        dst_crs: Target CRS
        resampling: Resampling method

    Returns:
        Tuple of (output_path, min_dn, max_dn, statistics)
    """
    logger.info(f"[raster_utils] Processing {len(tiff_urls)} TIFF files")

    masked_datasets = []
    meta_template = None

    for url in tiff_urls:
        try:
            dataset, meta_template = mask_tiff(
                url=url,
                polygon_gdf=polygon_gdf,
                null_values=null_values,
                nodata_value=nodata_value,
                meta_template=meta_template,
            )
            masked_datasets.append(dataset)
            logger.info(f"[raster_utils] Successfully masked: {url}")
        except Exception as e:
            logger.warning(f"[raster_utils] Skipping {url} due to error: {e}")

    if not masked_datasets:
        logger.error("[raster_utils] No masked datasets available to merge.")
        raise ValueError("Failed to process any TIFF files")

    # Merge datasets
    mosaic, out_trans = merge_datasets(masked_datasets)

    # Reproject
    dst_mosaic, out_meta, min_dn, max_dn = reproject_mosaic(
        mosaic=mosaic,
        out_trans=out_trans,
        null_values=null_values,
        nodata_value=nodata_value,
        meta_template=meta_template,
        dst_crs=dst_crs,
        resampling=resampling,
    )

    # Save to file
    save_raster_to_file(dst_mosaic, out_meta, output_path)

    # Calculate statistics
    valid_data = dst_mosaic[dst_mosaic != nodata_value]
    statistics = {
        "total_pixels": int(dst_mosaic.size),
        "valid_pixels": int(valid_data.size),
        "min_value": float(min_dn),
        "max_value": float(max_dn),
        "bounds": list(rasterio.transform.array_bounds(
            dst_mosaic.shape[1], dst_mosaic.shape[2], out_meta["transform"]
        )),
    }

    # For flood depth rasters (categorical 1-5)
    if max_dn <= 5:
        depth_counts = {}
        depth_labels = {
            1: "0-0.5m",
            2: "0.5-1m",
            3: "1-2m",
            4: "2-5m",
            5: ">5m"
        }
        for class_val in range(1, 6):
            count = int(np.sum(dst_mosaic == class_val))
            if count > 0:
                depth_counts[depth_labels[class_val]] = count
        statistics["depth_distribution"] = depth_counts

        # Risk distribution
        high_risk = int(np.sum((dst_mosaic >= 3) & (dst_mosaic <= 5)))
        medium_risk = int(np.sum(dst_mosaic == 2))
        low_risk = int(np.sum(dst_mosaic == 1))
        total_flood = high_risk + medium_risk + low_risk
        if total_flood > 0:
            statistics["risk_distribution"] = {
                "high": high_risk,
                "high_pct": round(high_risk / total_flood * 100, 1),
                "medium": medium_risk,
                "medium_pct": round(medium_risk / total_flood * 100, 1),
                "low": low_risk,
                "low_pct": round(low_risk / total_flood * 100, 1),
            }

    logger.info(f"[raster_utils] Successfully processed raster data")
    return output_path, min_dn, max_dn, statistics


def raster_to_geodataframe(
    raster_array: np.ndarray,
    transform: Affine,
    nodata_value: float,
    crs,
) -> gpd.GeoDataFrame:
    """
    Convert a raster array to a GeoDataFrame with polygon geometries.

    Args:
        raster_array: Raster data array
        transform: Affine transform
        nodata_value: Nodata value to exclude
        crs: Coordinate reference system

    Returns:
        GeoDataFrame with DN values and geometries
    """
    logger.debug("[raster_utils] Converting raster array to GeoDataFrame")

    raster_array = raster_array.astype(np.float32)
    mask_arr = raster_array[0] != nodata_value

    geoms = []
    values = []

    for geom, value in shapes(raster_array[0], mask=mask_arr, transform=transform):
        geoms.append(shape(geom))
        values.append(value)

    gdf = gpd.GeoDataFrame({"DN": values, "geometry": geoms}, crs=crs)
    logger.debug(f"[raster_utils] Created GeoDataFrame with {len(gdf)} features")
    return gdf
