"""Copernicus GloFAS Flood Hazard Data Service.

This service downloads and processes flood hazard data from the Copernicus
Emergency Management Service (CEMS) GloFAS dataset.

Data source: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/

Adapted from geoforge_services for RiskScoop AI.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.warp import Resampling
from shapely.geometry import box

from services.raster_utils import (
    mask_tiff,
    mask_tiff_from_source,
    merge_datasets,
    reproject_mosaic,
    save_raster_to_file,
)

logger = logging.getLogger(__name__)

# Configure GDAL for optimal COG access with HTTP caching
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '100000000'  # 100MB cache per file
os.environ['GDAL_CACHEMAX'] = '512'  # 512MB block cache
os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
os.environ['GDAL_HTTP_MULTIPLEX'] = 'YES'
os.environ['CPL_VSIL_CURL_CHUNK_SIZE'] = '1048576'  # 1MB chunks
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'


class CopernicusGLOFAS:
    """
    Data source class for Copernicus GloFAS flood hazard data.
    Reads tile extents from a GeoJSON file and builds TIFF URLs using the FTP base URL.
    """

    # Available return periods
    RETURN_PERIODS = ["RP10", "RP20", "RP50", "RP100", "RP200", "RP500"]

    # Depth classes for reclassified data
    DEPTH_CLASSES = {
        0: "No data",
        1: "0-0.5m",
        2: "0.5-1m",
        3: "1-2m",
        4: "2-5m",
        5: ">5m"
    }

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the GloFAS service.

        Args:
            output_dir: Directory for output files
        """
        base_dir = os.path.dirname(__file__)
        self.ftp_base_url = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard"
        self.tile_extents_path = os.path.join(base_dir, "tile_extents.geojson")

        # Nodata handling: source uses 255, we use 0 for output
        self.null_values = [255]  # Replace source nodata (255)
        self.no_data_value = 0  # Use 0 as nodata in output

        self.output_dir = Path(output_dir) if output_dir else Path("output/flood")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[copernicus_glofas] Initialized with tile extents: {self.tile_extents_path}")

    def get_tile_features(self) -> gpd.GeoDataFrame:
        """
        Reads the tile extents from the GeoJSON file and returns them as a GeoDataFrame.

        Returns:
            GeoDataFrame with tile extents in EPSG:4326
        """
        gdf = gpd.read_file(self.tile_extents_path)
        return gdf.to_crs("EPSG:4326")

    def get_intersecting_tile_features(
        self, polygon_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Returns only those tile features that intersect with the provided geometries.
        Uses spatial join to handle points, lines, and polygons correctly.

        Args:
            polygon_gdf: GeoDataFrame with AOI geometries

        Returns:
            GeoDataFrame with intersecting tiles
        """
        tiles_gdf = self.get_tile_features()

        # Normalize CRS
        if polygon_gdf.crs is not None and tiles_gdf.crs is not None:
            if polygon_gdf.crs != tiles_gdf.crs:
                polygon_gdf = polygon_gdf.to_crs(tiles_gdf.crs)

        # Use spatial join - works for all geometry types
        intersecting = gpd.sjoin(
            tiles_gdf,
            polygon_gdf,
            how="inner",
            predicate="intersects"
        )

        # Remove duplicate tiles
        intersecting = intersecting.drop_duplicates(subset=["id"])

        logger.info(f"[copernicus_glofas] Found {len(intersecting)} intersecting tile(s)")
        return intersecting

    def get_tiff_urls_from_features(
        self,
        tile_features: gpd.GeoDataFrame,
        return_period: str = "RP10"
    ) -> List[str]:
        """
        Constructs TIFF URLs from the tile features using the expected filename pattern.

        Args:
            tile_features: GeoDataFrame with tile metadata
            return_period: Return period (RP10, RP20, etc.)

        Returns:
            List of TIFF URLs
        """
        urls: List[str] = []

        if len(tile_features) == 0:
            logger.warning("[copernicus_glofas] No tile features provided")
            return urls

        for idx, row in tile_features.iterrows():
            tile_id = row.get("id")
            # Handle various column name variations after spatial join
            tile_name = row.get("name") or row.get("name_1") or row.get("name_left")

            if tile_id is None or tile_name is None:
                logger.warning(f"[copernicus_glofas] Skipping tile at index {idx}: id={tile_id}, name={tile_name}")
                continue

            filename = f"ID{tile_id}_{tile_name}_{return_period}_depth_reclass.tif"
            url = f"{self.ftp_base_url}/{return_period}/{filename}"
            urls.append(url)

        logger.debug(f"[copernicus_glofas] Generated {len(urls)} TIFF URL(s)")
        return urls

    def get_flood_hazard(
        self,
        aoi_gdf: gpd.GeoDataFrame,
        return_period: str = "RP10",
        output_path: Optional[Path] = None,
    ) -> Tuple[Optional[Path], dict]:
        """
        Get flood hazard data for an AOI.

        This method:
        1. Finds intersecting tiles
        2. Downloads and masks each tile
        3. Merges and reprojects the result
        4. Saves to output file

        Args:
            aoi_gdf: GeoDataFrame with the area of interest
            return_period: Return period (RP10, RP20, RP50, RP100, RP200, RP500)
            output_path: Optional path for output file

        Returns:
            Tuple of (output path, statistics dict)
        """
        if return_period not in self.RETURN_PERIODS:
            raise ValueError(f"Invalid return period. Must be one of: {self.RETURN_PERIODS}")

        logger.info(f"[copernicus_glofas] Getting flood hazard for {return_period}")

        # Find intersecting tiles
        intersecting_tiles = self.get_intersecting_tile_features(aoi_gdf)

        if intersecting_tiles.empty:
            logger.warning("[copernicus_glofas] No tiles found for AOI")
            return None, {"error": "No tiles found for this area"}

        # Build and deduplicate TIFF URLs
        tiff_urls_raw = self.get_tiff_urls_from_features(intersecting_tiles, return_period)
        tiff_urls = list(set(tiff_urls_raw))  # Deduplicate

        duplicates_removed = len(tiff_urls_raw) - len(tiff_urls)
        if duplicates_removed > 0:
            logger.info(f"[copernicus_glofas] Removed {duplicates_removed} duplicate tile URL(s)")

        logger.info(f"[copernicus_glofas] Processing {len(tiff_urls)} unique tile(s)")

        # Generate output path if not provided
        if output_path is None:
            output_id = str(uuid.uuid4())[:8]
            output_path = self.output_dir / f"flood_{return_period}_{output_id}.tif"

        try:
            # Process tiles: mask, merge, reproject, save
            result_path, statistics = self._process_tiles(
                tiff_urls=tiff_urls,
                aoi_gdf=aoi_gdf,
                output_path=output_path,
            )

            return result_path, statistics

        except Exception as e:
            logger.error(f"[copernicus_glofas] Error processing tiles: {e}")
            return None, {"error": str(e)}

    def _process_tiles(
        self,
        tiff_urls: List[str],
        aoi_gdf: gpd.GeoDataFrame,
        output_path: Path,
    ) -> Tuple[Path, dict]:
        """
        Process tiles: mask each with AOI, merge, reproject, and save.

        Args:
            tiff_urls: List of TIFF URLs to process
            aoi_gdf: AOI for masking
            output_path: Path for output file

        Returns:
            Tuple of (output path, statistics)
        """
        masked_datasets = []
        meta_template = None

        for url in tiff_urls:
            try:
                logger.info(f"[copernicus_glofas] Processing: {url}")

                # Open tile and mask
                with rasterio.open(url) as src:
                    if meta_template is None:
                        meta_template = src.meta.copy()

                    dataset, meta_template = mask_tiff_from_source(
                        src=src,
                        polygon_gdf=aoi_gdf,
                        null_values=self.null_values,
                        nodata_value=self.no_data_value,
                        meta_template=meta_template,
                        url=url,
                    )
                    masked_datasets.append(dataset)
                    logger.info(f"[copernicus_glofas] Successfully masked tile")

            except Exception as e:
                logger.warning(f"[copernicus_glofas] Skipping {url} due to error: {e}")
                continue

        if not masked_datasets:
            raise ValueError("Failed to process any TIFF files")

        logger.info(f"[copernicus_glofas] Successfully masked {len(masked_datasets)} tile(s)")

        # Merge datasets
        mosaic, out_trans = merge_datasets(masked_datasets)
        logger.info(f"[copernicus_glofas] Merged mosaic shape: {mosaic.shape}")

        # Reproject to EPSG:4326 for web display
        dst_mosaic, out_meta, min_dn, max_dn = reproject_mosaic(
            mosaic=mosaic,
            out_trans=out_trans,
            meta_template=meta_template,
            null_values=self.null_values,
            nodata_value=self.no_data_value,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,  # Use nearest for categorical data
        )

        # Save to file
        save_raster_to_file(dst_mosaic, out_meta, output_path)

        # Calculate statistics
        statistics = self._calculate_statistics(dst_mosaic, out_meta)
        statistics["output_path"] = str(output_path)

        logger.info(f"[copernicus_glofas] Created flood hazard raster: {output_path}")
        return output_path, statistics

    def _calculate_statistics(self, raster: np.ndarray, meta: dict) -> dict:
        """
        Calculate statistics for flood hazard raster.

        Args:
            raster: Raster data array
            meta: Raster metadata

        Returns:
            Dictionary with statistics
        """
        # Get valid data (exclude nodata)
        valid_data = raster[(raster != self.no_data_value) & (raster >= 1) & (raster <= 5)]

        stats = {
            "total_pixels": int(raster.size),
            "flood_pixels": int(valid_data.size),
            "flood_percentage": round(valid_data.size / raster.size * 100, 2) if raster.size > 0 else 0,
            "bounds": list(rasterio.transform.array_bounds(
                raster.shape[1] if raster.ndim == 3 else raster.shape[0],
                raster.shape[2] if raster.ndim == 3 else raster.shape[1],
                meta["transform"]
            )),
            "crs": str(meta["crs"]),
        }

        if valid_data.size > 0:
            # Depth class distribution
            raster_2d = raster[0] if raster.ndim == 3 else raster
            depth_counts = {}
            for class_val, class_name in self.DEPTH_CLASSES.items():
                if class_val > 0:  # Skip "No data" class
                    count = int(np.sum(raster_2d == class_val))
                    if count > 0:
                        depth_counts[class_name] = count
            stats["depth_classes"] = depth_counts

            # Risk distribution
            high_risk = int(np.sum((raster_2d >= 3) & (raster_2d <= 5)))  # 1m+ depth
            medium_risk = int(np.sum(raster_2d == 2))  # 0.5-1m
            low_risk = int(np.sum(raster_2d == 1))  # 0-0.5m
            total_flood = high_risk + medium_risk + low_risk

            if total_flood > 0:
                stats["risk_distribution"] = {
                    "high": high_risk,
                    "high_pct": round(high_risk / total_flood * 100, 1),
                    "medium": medium_risk,
                    "medium_pct": round(medium_risk / total_flood * 100, 1),
                    "low": low_risk,
                    "low_pct": round(low_risk / total_flood * 100, 1),
                }

        return stats

    def get_bounds_geojson(self, raster_path: Path) -> dict:
        """
        Get raster bounds as GeoJSON for map overlay positioning.

        Args:
            raster_path: Path to raster file

        Returns:
            Dictionary with bounds coordinates for Mapbox
        """
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            return {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [bounds.left, bounds.bottom],
                        [bounds.right, bounds.bottom],
                        [bounds.right, bounds.top],
                        [bounds.left, bounds.top],
                        [bounds.left, bounds.bottom]
                    ]]
                },
                "properties": {
                    "bounds": {
                        "west": bounds.left,
                        "south": bounds.bottom,
                        "east": bounds.right,
                        "north": bounds.top
                    },
                    "coordinates": [
                        [bounds.left, bounds.top],     # top-left
                        [bounds.right, bounds.top],    # top-right
                        [bounds.right, bounds.bottom], # bottom-right
                        [bounds.left, bounds.bottom]   # bottom-left
                    ]
                }
            }
