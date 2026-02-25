"""GloFAS Flood Forecast Service.

This service provides access to GloFAS (Global Flood Awareness System) flood data.
It downloads real flood hazard data from Copernicus CEMS.

The service generates flood risk assessments based on flood depth data.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

from services.copernicus_glofas import CopernicusGLOFAS

logger = logging.getLogger(__name__)


class GloFASService:
    """Service for fetching and processing GloFAS flood forecast data."""

    # Risk thresholds based on flood depth classes
    # Reclassified data uses these classes:
    # 1: 0-0.5m, 2: 0.5-1m, 3: 1-2m, 4: 2-5m, 5: >5m
    RISK_THRESHOLDS = {
        "high": 3,      # 1m+ depth (classes 3, 4, 5)
        "medium": 2,    # 0.5-1m depth (class 2)
        "low": 1,       # 0-0.5m depth (class 1)
        "minimal": 0    # No flood
    }

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the GloFAS service."""
        logger.info("[glofas_service] Initializing GloFAS Service")
        self.output_dir = Path(output_dir) if output_dir else Path("output/flood")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.copernicus = CopernicusGLOFAS(output_dir=str(self.output_dir))

    def get_flood_forecast(
        self,
        aoi_gdf: gpd.GeoDataFrame,
        return_period: str = "RP10",
    ) -> gpd.GeoDataFrame:
        """Get flood forecast data for an Area of Interest.

        Args:
            aoi_gdf: GeoDataFrame with the AOI geometry
            return_period: Return period for flood hazard (RP10, RP20, etc.)

        Returns:
            GeoDataFrame with flood forecast points containing:
            - geometry: Point locations
            - flood_depth_class: Flood depth class (1-5)
            - risk_level: Flood risk classification (minimal/low/medium/high)
            - timestamp: Forecast timestamp
            - data_source: Source of the data (copernicus)
            - raster_path: Path to the TIFF file
        """
        logger.info(f"[glofas_service] Getting flood forecast for AOI, return period: {return_period}")

        # Get bounding box from AOI
        bounds = aoi_gdf.total_bounds  # [minx, miny, maxx, maxy]

        # Fetch real Copernicus data
        gdf, raster_path = self._fetch_copernicus_data(aoi_gdf, return_period)

        if gdf is not None and not gdf.empty:
            return gdf

        # If Copernicus data fails, return empty GeoDataFrame
        logger.warning("[glofas_service] Failed to get Copernicus data")
        return gpd.GeoDataFrame()

    def _fetch_copernicus_data(
        self,
        aoi_gdf: gpd.GeoDataFrame,
        return_period: str = "RP10"
    ) -> tuple[Optional[gpd.GeoDataFrame], Optional[Path]]:
        """Fetch flood data from Copernicus GloFAS.

        Args:
            aoi_gdf: GeoDataFrame with AOI
            return_period: Return period (RP10, RP20, etc.)

        Returns:
            Tuple of (GeoDataFrame with flood data points, raster path)
        """
        try:
            # Download and process flood hazard data
            result_path, stats = self.copernicus.get_flood_hazard(
                aoi_gdf=aoi_gdf,
                return_period=return_period,
            )

            if result_path is None:
                return None, None

            # Convert raster to points for GeoDataFrame
            gdf = self._raster_to_points(result_path)

            if gdf.empty:
                return None, None

            # Add metadata
            gdf['timestamp'] = datetime.now().isoformat()
            gdf['data_source'] = 'copernicus'
            gdf['return_period'] = return_period
            gdf['raster_path'] = str(result_path)

            logger.info(f"[glofas_service] Retrieved {len(gdf)} flood points from Copernicus")
            return gdf, result_path

        except Exception as e:
            logger.error(f"[glofas_service] Error fetching Copernicus data: {e}")
            return None, None

    def _raster_to_points(self, raster_path: Path) -> gpd.GeoDataFrame:
        """Convert flood raster to point GeoDataFrame.

        Args:
            raster_path: Path to flood hazard raster

        Returns:
            GeoDataFrame with flood points
        """
        points = []
        data_list = []

        # Depth class labels for tooltips
        depth_labels = {
            1: "0-0.5m",
            2: "0.5-1m",
            3: "1-2m",
            4: "2-5m",
            5: ">5m"
        }

        with rasterio.open(raster_path) as src:
            data = src.read(1)
            transform = src.transform
            nodata = src.nodata if src.nodata is not None else 0

            # Full resolution - iterate all pixels with flood data
            rows, cols = data.shape

            for row in range(rows):
                for col in range(cols):
                    value = data[row, col]

                    # Skip nodata (0) and values outside valid range (1-5)
                    if value >= 1 and value <= 5:
                        # Convert pixel to coordinates
                        x, y = rasterio.transform.xy(transform, row, col)

                        depth_class = int(value)
                        risk_level = self._classify_risk(depth_class)

                        points.append(Point(x, y))
                        data_list.append({
                            'flood_depth_class': depth_class,
                            'flood_depth': depth_labels.get(depth_class, 'Unknown'),
                            'risk_level': risk_level,
                            'description': f"Flood depth: {depth_labels.get(depth_class, 'Unknown')}, Risk: {risk_level.capitalize()}"
                        })

        if not points:
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(data_list, geometry=points, crs="EPSG:4326")
        logger.info(f"[glofas_service] Converted raster to {len(gdf)} points (full resolution)")
        return gdf

    def _classify_risk(self, depth_class: int) -> str:
        """Classify flood risk based on depth class.

        Args:
            depth_class: Flood depth class (1-5)

        Returns:
            Risk level string
        """
        if depth_class >= self.RISK_THRESHOLDS["high"]:
            return "high"
        elif depth_class >= self.RISK_THRESHOLDS["medium"]:
            return "medium"
        elif depth_class >= self.RISK_THRESHOLDS["low"]:
            return "low"
        else:
            return "minimal"

    def get_risk_summary(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Generate a summary of flood risk for a GeoDataFrame.

        Args:
            gdf: GeoDataFrame with flood forecast data

        Returns:
            Dictionary with risk statistics
        """
        if gdf.empty:
            return {"error": "No data available"}

        summary = {
            "total_points": len(gdf),
            "timestamp": datetime.now().isoformat(),
            "risk_distribution": gdf['risk_level'].value_counts().to_dict(),
            "high_risk_count": len(gdf[gdf['risk_level'] == 'high']),
            "data_source": gdf['data_source'].iloc[0] if 'data_source' in gdf.columns else 'unknown'
        }

        # Calculate percentage of area at risk
        total = len(gdf)
        if total > 0:
            summary["risk_percentages"] = {
                "high": round(len(gdf[gdf['risk_level'] == 'high']) / total * 100, 1),
                "medium": round(len(gdf[gdf['risk_level'] == 'medium']) / total * 100, 1),
                "low": round(len(gdf[gdf['risk_level'] == 'low']) / total * 100, 1),
                "minimal": round(len(gdf[gdf['risk_level'] == 'minimal']) / total * 100, 1)
            }

        # Add depth class distribution if available
        if 'flood_depth_class' in gdf.columns:
            depth_counts = gdf['flood_depth_class'].value_counts().to_dict()
            depth_labels = {
                1: "0-0.5m",
                2: "0.5-1m",
                3: "1-2m",
                4: "2-5m",
                5: ">5m"
            }
            summary["depth_distribution"] = {
                depth_labels.get(k, str(k)): v
                for k, v in depth_counts.items()
            }

        return summary

    def get_raster_layer_info(self, gdf: gpd.GeoDataFrame) -> Optional[Dict]:
        """Get raster layer info for map display.

        Args:
            gdf: GeoDataFrame with flood data (must have raster_path)

        Returns:
            Dictionary with raster layer info for frontend
        """
        if gdf.empty or 'raster_path' not in gdf.columns:
            return None

        raster_path = Path(gdf['raster_path'].iloc[0])

        if not raster_path.exists():
            return None

        # Get bounds for map overlay
        bounds_info = self.copernicus.get_bounds_geojson(raster_path)

        return {
            "type": "raster",
            "tiff_path": str(raster_path),
            "bounds": bounds_info["properties"]["bounds"],
            "coordinates": bounds_info["properties"]["coordinates"]
        }
