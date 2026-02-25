"""GloFAS Flood Forecast Tool.

This tool retrieves flood hazard data from Copernicus CEMS GloFAS dataset
for a specified Area of Interest (AOI). It downloads real flood depth data
from the European Commission's Joint Research Centre.

Usage Flow:
1. User specifies an AOI layer (created with get_division)
2. Tool downloads flood hazard TIFF tiles from Copernicus
3. Returns flood risk points with depth classes and risk levels
4. Data can be visualized on map as raster overlay or points

Data Source:
- Copernicus Emergency Management Service (CEMS)
- Global Flood Awareness System (GloFAS)
- https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/
"""

import logging
from agno.tools import tool
from agno.run import RunContext

from services.glofas_service import GloFASService
from services.layer_service import LayerService

logger = logging.getLogger(__name__)

glofas_service = GloFASService()
layer_service = LayerService()


@tool
def get_flood_forecast(
    run_context: RunContext,
    aoi_layer_name: str,
    output_layer_name: str,
) -> str:
    """Get flood hazard data from Copernicus GloFAS for an Area of Interest.

    This tool retrieves flood hazard data from the Copernicus CEMS GloFAS dataset.
    It downloads actual flood depth data (TIFF files) and converts them to risk points.

    Flood depth classes:
    - Class 1: 0-0.5m depth (Low risk)
    - Class 2: 0.5-1m depth (Medium risk)
    - Class 3: 1-2m depth (High risk)
    - Class 4: 2-5m depth (High risk)
    - Class 5: >5m depth (High risk)

    Args:
        session_state: The session state dictionary containing layer references.
        aoi_layer_name: Name of the AOI layer in session state (from get_division).
        output_layer_name: Name for the output flood forecast layer.

    Returns:
        str: Summary of flood hazard results including depth distribution and risk levels.

    Example:
        >>> # First create an AOI
        >>> get_division(session_state, "Geneva, Switzerland", "aoi")
        >>> # Then get flood hazard data
        >>> get_flood_forecast(session_state, "aoi", "flood_risk")
        "Retrieved flood hazard data: 1500 points from Copernicus. High risk: 150 (10.0%), ..."

    Note:
        - Requires an AOI layer created with get_division
        - Downloads real flood hazard data from Copernicus CEMS
        - Results can be overlaid with buildings/infrastructure for impact analysis
    """
    session_state = run_context.session_state
    if session_state is None:
        session_state = {}

    logger.info(
        f"[get_flood_tool] Getting flood hazard for AOI: {aoi_layer_name}"
    )

    # Get AOI layer UUID from session state
    if "layers" not in session_state or aoi_layer_name not in session_state["layers"]:
        return f"Error: AOI layer '{aoi_layer_name}' not found. Create it first with get_division."

    aoi_uuid = session_state["layers"][aoi_layer_name]

    # Load AOI layer
    aoi_gdf = layer_service.load_layer(aoi_uuid)
    if aoi_gdf.empty:
        return f"Error: Could not load AOI layer '{aoi_layer_name}'."

    # Get flood forecast
    try:
        gdf = glofas_service.get_flood_forecast(
            aoi_gdf=aoi_gdf,
        )
    except Exception as e:
        logger.error(f"[get_flood_tool] Error getting flood hazard: {e}")
        return f"Error retrieving flood hazard data: {str(e)}"

    if gdf.empty:
        return "No flood hazard data available for this area. The area may not have significant flood risk."

    # Save the layer
    layer_service.save_layer(gdf, session_state, output_layer_name)

    # Check for raster data and save info
    raster_info = glofas_service.get_raster_layer_info(gdf)
    if raster_info:
        # Store raster info in session for frontend
        if "raster_layers" not in session_state:
            session_state["raster_layers"] = {}
        session_state["raster_layers"][output_layer_name] = raster_info

    # Generate summary
    summary = glofas_service.get_risk_summary(gdf)

    # Format response
    response_parts = [
        f"Retrieved flood hazard data: {summary['total_points']} points.",
        f"Data source: {summary['data_source']}",
    ]

    if "risk_percentages" in summary:
        for level in ["high", "medium", "low", "minimal"]:
            count = summary["risk_distribution"].get(level, 0)
            pct = summary["risk_percentages"].get(level, 0)
            response_parts.append(f"  - {level.capitalize()}: {count} ({pct}%)")

    if "depth_distribution" in summary:
        response_parts.extend(["", "Flood Depth Distribution:"])
        for depth_class, count in summary["depth_distribution"].items():
            response_parts.append(f"  - {depth_class}: {count} points")

    response_parts.extend(["", f"Saved as layer: '{output_layer_name}'"])

    if summary["high_risk_count"] > 0:
        response_parts.append(
            f"\n⚠️ WARNING: {summary['high_risk_count']} locations with HIGH flood risk (>1m depth)!"
        )

    if raster_info:
        response_parts.append(f"\nRaster layer available for map overlay.")

    return "\n".join(response_parts)
