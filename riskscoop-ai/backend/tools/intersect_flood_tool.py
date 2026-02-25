"""Flood Intersection Tool.

This tool performs spatial intersection between features (buildings, infrastructure, etc.)
and flood hazard zones to identify which features are at risk of flooding.

Usage Flow:
1. User has a features layer (e.g., hospitals from get_overture_data)
2. User has a flood layer (from get_flood_forecast)
3. This tool intersects them to find features in flood zones
4. Returns features with flood risk attributes attached
"""

import json
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from agno.tools import tool
from shapely.geometry import Point

from services.layer_service import LayerService

logger = logging.getLogger(__name__)

layer_service = LayerService()

# Directory for storing flood statistics JSON
FLOOD_STATS_DIR = Path("output/flood_stats")
FLOOD_STATS_DIR.mkdir(parents=True, exist_ok=True)


@tool
def intersect_flood_zones(
    session_state: dict,
    features_layer_name: str,
    flood_layer_name: str,
    output_layer_name: str,
    buffer_meters: float = 100.0,
    min_risk_level: str = "low",
) -> str:
    """Find features that are in flood risk zones.

    This tool performs spatial intersection between a features layer (buildings,
    infrastructure, POIs, etc.) and a flood hazard layer to identify which
    features are at risk of flooding.

    Each feature in the output will have flood risk attributes:
    - flood_risk_level: The highest risk level the feature intersects (high/medium/low)
    - flood_depth_class: The maximum flood depth class (1-5)
    - flood_depth_range: Human-readable depth range (e.g., "1-2m")

    Args:
        session_state: The session state dictionary containing layer references.
        features_layer_name: Name of the features layer (from get_overture_data).
        flood_layer_name: Name of the flood layer (from get_flood_forecast).
        output_layer_name: Name for the output layer with flood-affected features.
        buffer_meters: Buffer distance in meters around features for intersection.
            Default 100m. Use larger values for area-based risk assessment.
        min_risk_level: Minimum risk level to include ("low", "medium", "high").
            Default "low" includes all flood zones.

    Returns:
        str: Summary of intersection results including count and risk breakdown.

    Example:
        >>> # First get hospitals and flood data
        >>> get_division(session_state, "Geneva, Switzerland", "aoi")
        >>> get_overture_data(session_state, "aoi", "SELECT * FROM overture_buildings WHERE class = 'hospital'", "hospitals")
        >>> get_flood_forecast(session_state, "aoi", "flood_risk")
        >>> # Find hospitals in flood zones
        >>> intersect_flood_zones(session_state, "hospitals", "flood_risk", "hospitals_at_risk")
        "Found 5 features in flood zones: 2 high risk, 1 medium risk, 2 low risk"

    Note:
        - Requires both a features layer and a flood layer in session state
        - Buffer helps catch features near flood zones (100m default)
        - Only features intersecting flood zones are included in output
        - Use with any feature type: buildings, infrastructure, places, etc.
    """
    logger.info(f"[intersect_flood_tool] Intersecting {features_layer_name} with {flood_layer_name}")

    # Validate risk level
    valid_risk_levels = ["low", "medium", "high"]
    if min_risk_level not in valid_risk_levels:
        return f"Error: Invalid min_risk_level '{min_risk_level}'. Must be one of: {', '.join(valid_risk_levels)}"

    # Get layers from session state
    if "layers" not in session_state:
        return "Error: No layers found in session state."

    if features_layer_name not in session_state["layers"]:
        return f"Error: Features layer '{features_layer_name}' not found. Create it first with get_overture_data."

    if flood_layer_name not in session_state["layers"]:
        return f"Error: Flood layer '{flood_layer_name}' not found. Create it first with get_flood_forecast."

    # Load layers
    features_uuid = session_state["layers"][features_layer_name]
    flood_uuid = session_state["layers"][flood_layer_name]

    features_gdf = layer_service.load_layer(features_uuid)
    flood_gdf = layer_service.load_layer(flood_uuid)

    if features_gdf.empty:
        return f"Error: Features layer '{features_layer_name}' is empty."

    if flood_gdf.empty:
        return f"Error: Flood layer '{flood_layer_name}' is empty."

    logger.info(f"[intersect_flood_tool] Features: {len(features_gdf)}, Flood points: {len(flood_gdf)}")

    # Filter flood data by minimum risk level
    risk_order = {"low": 1, "medium": 2, "high": 3}
    min_risk_value = risk_order[min_risk_level]

    if "risk_level" in flood_gdf.columns:
        flood_gdf = flood_gdf[flood_gdf["risk_level"].map(risk_order) >= min_risk_value].copy()

    if flood_gdf.empty:
        return f"No flood zones found with risk level >= '{min_risk_level}'."

    # Perform spatial intersection
    try:
        result_gdf = _spatial_intersect(
            features_gdf=features_gdf,
            flood_gdf=flood_gdf,
            buffer_meters=buffer_meters
        )
    except Exception as e:
        logger.error(f"[intersect_flood_tool] Intersection error: {e}")
        return f"Error during spatial intersection: {str(e)}"

    if result_gdf.empty:
        return f"No features from '{features_layer_name}' found in flood zones (risk >= {min_risk_level})."

    # Save the result layer
    layer_service.save_layer(result_gdf, session_state, output_layer_name)

    # Generate summary with chart data
    summary = _generate_summary(result_gdf, len(features_gdf))

    # Generate detailed statistics for charts
    chart_data = _generate_chart_data(result_gdf, features_gdf, features_layer_name, flood_layer_name)

    # Store chart data in session state for frontend retrieval
    if "flood_statistics" not in session_state:
        session_state["flood_statistics"] = {}
    session_state["flood_statistics"][output_layer_name] = chart_data

    # Save chart data to JSON file for API access
    stats_file = FLOOD_STATS_DIR / f"{output_layer_name}.json"
    with open(stats_file, "w") as f:
        json.dump(chart_data, f, indent=2)
    logger.info(f"[intersect_flood_tool] Saved chart data to {stats_file}")

    # Format response with embedded chart data marker
    response_parts = [
        f"## Flood Risk Analysis: {features_layer_name}",
        "",
        f"**{summary['total_at_risk']}** out of **{summary['total_features']}** features ({summary['percentage_at_risk']}%) are in flood risk zones.",
        "",
        "### Risk Level Breakdown:",
    ]

    for level in ["high", "medium", "low"]:
        count = summary["risk_counts"].get(level, 0)
        if count > 0:
            emoji = "🔴" if level == "high" else ("🟠" if level == "medium" else "🟡")
            response_parts.append(f"- {emoji} **{level.capitalize()}** risk: {count} features")

    if "depth_distribution" in summary:
        response_parts.extend([
            "",
            "### Flood Depth Distribution:"
        ])
        depth_order = ["0-0.5m", "0.5-1m", "1-2m", "2-5m", ">5m"]
        for depth in depth_order:
            count = summary["depth_distribution"].get(depth, 0)
            if count > 0:
                response_parts.append(f"- **{depth}**: {count} features")

    response_parts.extend([
        "",
        f"📊 **[View Detailed Charts](flood_stats:{output_layer_name})**",
        "",
        f"_Layer saved as: `{output_layer_name}`_"
    ])

    if summary["risk_counts"].get("high", 0) > 0:
        response_parts.append(
            f"\n⚠️ **CRITICAL WARNING**: {summary['risk_counts']['high']} features are in **HIGH** flood risk zones (>1m depth)!"
        )

    return "\n".join(response_parts)


def _spatial_intersect(
    features_gdf: gpd.GeoDataFrame,
    flood_gdf: gpd.GeoDataFrame,
    buffer_meters: float
) -> gpd.GeoDataFrame:
    """Perform spatial intersection between features and flood zones.

    Args:
        features_gdf: GeoDataFrame with features to check
        flood_gdf: GeoDataFrame with flood risk points
        buffer_meters: Buffer distance around features

    Returns:
        GeoDataFrame with features that intersect flood zones, with risk attributes
    """
    # Ensure both GDFs have the same CRS
    if features_gdf.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(features_gdf.crs)

    # Convert buffer from meters to degrees (approximate)
    # At equator: 1 degree ~ 111km, at 45 lat: 1 degree ~ 78km for longitude
    # Use a rough average for mid-latitudes
    buffer_degrees = buffer_meters / 100000  # More generous buffer conversion

    # Create buffered features for intersection
    features_buffered = features_gdf.copy()
    # Suppress the CRS warning - we're aware and using approximate conversion
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features_buffered["geometry"] = features_gdf.geometry.buffer(buffer_degrees)

    # Spatial join to find intersecting flood points
    joined = gpd.sjoin(
        features_buffered,
        flood_gdf[["geometry", "risk_level", "flood_depth_class"]],
        how="inner",
        predicate="intersects"
    )

    if joined.empty:
        return gpd.GeoDataFrame()

    # Aggregate flood risk per feature (take max risk level)
    risk_order = {"low": 1, "medium": 2, "high": 3}
    risk_reverse = {1: "low", 2: "medium", 3: "high"}

    # Depth class descriptions
    depth_labels = {
        1: "0-0.5m",
        2: "0.5-1m",
        3: "1-2m",
        4: "2-5m",
        5: ">5m"
    }

    # Group by original feature index and aggregate
    aggregated = joined.groupby(joined.index).agg({
        "risk_level": lambda x: risk_reverse[max(risk_order.get(v, 0) for v in x)],
        "flood_depth_class": "max"
    }).reset_index()

    aggregated.columns = ["feature_idx", "flood_risk_level", "flood_depth_class"]

    # Merge back with original features (using original geometry, not buffered)
    result = features_gdf.loc[aggregated["feature_idx"]].copy()
    result["flood_risk_level"] = aggregated["flood_risk_level"].values
    result["flood_depth_class"] = aggregated["flood_depth_class"].values
    result["flood_depth_range"] = result["flood_depth_class"].map(depth_labels)

    return result.reset_index(drop=True)


def _generate_summary(result_gdf: gpd.GeoDataFrame, total_features: int) -> dict:
    """Generate summary statistics for intersection results.

    Args:
        result_gdf: GeoDataFrame with intersection results
        total_features: Total number of features before intersection

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_features": total_features,
        "total_at_risk": len(result_gdf),
        "percentage_at_risk": round(len(result_gdf) / total_features * 100, 1) if total_features > 0 else 0,
        "risk_counts": {}
    }

    if "flood_risk_level" in result_gdf.columns:
        summary["risk_counts"] = result_gdf["flood_risk_level"].value_counts().to_dict()

    if "flood_depth_range" in result_gdf.columns:
        summary["depth_distribution"] = result_gdf["flood_depth_range"].value_counts().to_dict()

    return summary


def _generate_chart_data(
    result_gdf: gpd.GeoDataFrame,
    features_gdf: gpd.GeoDataFrame,
    features_layer_name: str,
    flood_layer_name: str
) -> dict:
    """Generate chart data for frontend visualization.

    Args:
        result_gdf: GeoDataFrame with intersection results (features at risk)
        features_gdf: Original features GeoDataFrame (all features)
        features_layer_name: Name of the features layer
        flood_layer_name: Name of the flood layer

    Returns:
        Dictionary with chart data for pie and bar charts
    """
    total_features = len(features_gdf)
    at_risk_features = len(result_gdf)
    safe_features = total_features - at_risk_features

    # Risk level distribution
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    if "flood_risk_level" in result_gdf.columns:
        for level in ["high", "medium", "low"]:
            risk_counts[level] = int((result_gdf["flood_risk_level"] == level).sum())

    # Depth distribution
    depth_counts = {"0-0.5m": 0, "0.5-1m": 0, "1-2m": 0, "2-5m": 0, ">5m": 0}
    if "flood_depth_range" in result_gdf.columns:
        for depth in depth_counts.keys():
            depth_counts[depth] = int((result_gdf["flood_depth_range"] == depth).sum())

    # Feature type breakdown (if available)
    feature_type_at_risk = {}
    feature_type_field = None

    # Check for common type fields
    for field in ["class", "subtype", "primary_category", "type"]:
        if field in result_gdf.columns:
            feature_type_field = field
            break

    if feature_type_field:
        type_counts = result_gdf[feature_type_field].value_counts().to_dict()
        # Convert to int for JSON serialization
        feature_type_at_risk = {str(k): int(v) for k, v in type_counts.items() if k is not None and str(k) != "None"}

    return {
        "summary": {
            "total_features": total_features,
            "at_risk": at_risk_features,
            "safe": safe_features,
            "percentage_at_risk": round(at_risk_features / total_features * 100, 1) if total_features > 0 else 0
        },
        "risk_distribution": {
            "labels": ["High Risk", "Medium Risk", "Low Risk"],
            "values": [risk_counts["high"], risk_counts["medium"], risk_counts["low"]],
            "colors": ["#ef4444", "#f59e0b", "#fcd34d"],
            "backgroundColors": ["rgba(239, 68, 68, 0.8)", "rgba(245, 158, 11, 0.8)", "rgba(252, 211, 77, 0.8)"]
        },
        "depth_distribution": {
            "labels": list(depth_counts.keys()),
            "values": list(depth_counts.values()),
            "colors": ["#fef3c7", "#fcd34d", "#f59e0b", "#ef4444", "#b91c1c"],
            "backgroundColors": [
                "rgba(254, 243, 199, 0.8)",
                "rgba(252, 211, 77, 0.8)",
                "rgba(245, 158, 11, 0.8)",
                "rgba(239, 68, 68, 0.8)",
                "rgba(185, 28, 28, 0.8)"
            ]
        },
        "exposure_pie": {
            "labels": ["At Risk", "Safe"],
            "values": [at_risk_features, safe_features],
            "colors": ["#ef4444", "#10b981"],
            "backgroundColors": ["rgba(239, 68, 68, 0.8)", "rgba(16, 185, 129, 0.8)"]
        },
        "feature_types": feature_type_at_risk,
        "metadata": {
            "features_layer": features_layer_name,
            "flood_layer": flood_layer_name,
            "analysis_type": "flood_intersection"
        }
    }
