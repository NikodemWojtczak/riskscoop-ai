"""SQL Tool Utilities for Geospatial Data Processing.

This module provides utility functions for processing SQL queries with spatial
filtering using Area of Interest (AOI) layers stored via LayerService.
"""

import logging
import re
from typing import Any, Dict, Optional

import geopandas as gpd

from services.md_service import MDService
from services.layer_service import LayerService

logger = logging.getLogger(__name__)

_md_service = MDService()
_layer_service = LayerService()


def get_table_name_from_sql(sql_query: str) -> Optional[str]:
    """Extract the table name from a SQL SELECT query.

    Args:
        sql_query: The SQL query string to parse.

    Returns:
        The extracted table name, or None if not found.

    Example:
        >>> get_table_name_from_sql("SELECT * FROM overture_buildings WHERE class='building'")
        'overture_buildings'
    """
    cleaned_query = " ".join(sql_query.strip().split())
    match = re.search(r"\bfrom\s+([a-zA-Z0-9_.]+)", cleaned_query, re.IGNORECASE)
    if match:
        return match.group(1)
    logger.warning(f"[sql_tool_utils] Could not extract table name from SQL: {sql_query}")
    return None


def add_limit_to_sql(sql: str, limit: int) -> str:
    """Add a LIMIT clause to a SQL query if not already present.

    Args:
        sql: The SQL query string.
        limit: Maximum number of rows to return.

    Returns:
        SQL query with LIMIT clause added.
    """
    if re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {limit}"


def execute_sql_with_aoi(
    sql: str,
    aoi_layer_uuid: str,
    session_state: Optional[Dict[str, Any]] = None,
    output_layer_name: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Execute SQL query with spatial filtering using an AOI layer.

    This function loads the AOI layer, filters by bounding box first for
    performance, then performs actual geometry intersection to get only
    features that truly intersect with the AOI.

    Args:
        sql: SQL query to execute (should query a table with bbox columns).
        aoi_layer_uuid: UUID of the AOI layer file.
        session_state: Optional session state to save result layer.
        output_layer_name: Optional name for the output layer in session state.

    Returns:
        GeoDataFrame with query results intersected with AOI.

    Example:
        >>> gdf = execute_sql_with_aoi(
        ...     "SELECT * FROM overture_buildings",
        ...     aoi_uuid,
        ...     session_state,
        ...     "buildings_layer"
        ... )
    """
    logger.info(f"[sql_tool_utils] Executing SQL with AOI: {aoi_layer_uuid}")

    # Load AOI layer
    aoi_gdf = _layer_service.load_layer(aoi_layer_uuid)
    if aoi_gdf.empty:
        logger.warning("[sql_tool_utils] AOI layer is empty")
        return gpd.GeoDataFrame()

    # Get bounding box from AOI for initial filtering
    bounds = aoi_gdf.total_bounds  # [minx, miny, maxx, maxy]
    minx, miny, maxx, maxy = bounds

    logger.debug(f"[sql_tool_utils] AOI bounds: {bounds}")

    # Build spatial filter SQL
    table_name = get_table_name_from_sql(sql)
    if not table_name:
        raise ValueError("Could not extract table name from SQL query")

    # Add bounding box filter to SQL (coarse filter for performance)
    bbox_filter = (
        f"bbox.xmin >= {minx} AND bbox.ymin >= {miny} "
        f"AND bbox.xmax <= {maxx} AND bbox.ymax <= {maxy}"
    )

    # Check if SQL already has WHERE clause
    if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        filtered_sql = re.sub(
            r"\bWHERE\b",
            f"WHERE {bbox_filter} AND ",
            sql,
            count=1,
            flags=re.IGNORECASE
        )
    else:
        # Add WHERE clause before any ORDER BY, GROUP BY, or LIMIT
        match = re.search(
            r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT)\b",
            sql,
            re.IGNORECASE
        )
        if match:
            insert_pos = match.start()
            filtered_sql = f"{sql[:insert_pos]} WHERE {bbox_filter} {sql[insert_pos:]}"
        else:
            filtered_sql = f"{sql.rstrip().rstrip(';')} WHERE {bbox_filter}"

    # Wrap geometry column with ST_AsText to get WKT format
    # Replace "geometry" or "geometry," with "ST_AsText(geometry) as geometry"
    filtered_sql = re.sub(
        r'\bgeometry\b(?=\s*,|\s+FROM|\s*$)',
        'ST_AsText(geometry) as geometry',
        filtered_sql,
        flags=re.IGNORECASE
    )
    # Handle SELECT * - need to expand it
    if re.search(r'\bSELECT\s+\*\s+FROM\b', filtered_sql, re.IGNORECASE):
        # Replace SELECT * with explicit column selection including ST_AsText(geometry)
        filtered_sql = re.sub(
            r'\bSELECT\s+\*\s+FROM\s+(\w+)',
            r'SELECT ST_AsText(geometry) as geometry, * EXCLUDE (geometry) FROM \1',
            filtered_sql,
            flags=re.IGNORECASE
        )

    logger.debug(f"[sql_tool_utils] Filtered SQL: {filtered_sql}")

    # Execute query (bbox-filtered)
    result = _md_service.execute_query(filtered_sql)
    df = result.fetchdf()

    if df.empty:
        logger.warning("[sql_tool_utils] Query returned no results")
        return gpd.GeoDataFrame()

    # Convert to GeoDataFrame
    if "geometry" not in df.columns:
        logger.warning("[sql_tool_utils] No geometry column in results")
        return gpd.GeoDataFrame(df)

    from shapely import wkt

    def parse_geometry(geom):
        """Parse geometry from WKT string (ST_AsText output)."""
        if geom is None:
            return None
        if hasattr(geom, 'geom_type'):
            # Already a Shapely geometry
            return geom
        if isinstance(geom, str):
            try:
                return wkt.loads(geom)
            except Exception as e:
                logger.warning(f"[sql_tool_utils] Failed to parse WKT: {e}")
                return None
        logger.warning(f"[sql_tool_utils] Unexpected geometry type: {type(geom)}")
        return None

    df["geometry"] = df["geometry"].apply(parse_geometry)

    # Remove rows with None geometry
    df = df[df["geometry"].notna()]

    if df.empty:
        logger.warning("[sql_tool_utils] No valid geometries after parsing")
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    logger.info(f"[sql_tool_utils] Bbox filter returned {len(gdf)} features")

    # Perform actual intersection with AOI geometry
    # Union all AOI geometries into a single geometry for intersection
    aoi_union = aoi_gdf.geometry.union_all()

    # Filter to only features that intersect with AOI
    gdf = gdf[gdf.geometry.intersects(aoi_union)].copy()

    logger.info(f"[sql_tool_utils] After intersection: {len(gdf)} features")

    if gdf.empty:
        logger.warning("[sql_tool_utils] No features intersect with AOI")
        return gpd.GeoDataFrame()

    # Optionally save to layer service
    if session_state is not None and output_layer_name is not None:
        _layer_service.save_layer(gdf, session_state, output_layer_name)
        logger.info(f"[sql_tool_utils] Saved result as layer: {output_layer_name}")

    return gdf


def execute_sql(sql: str) -> gpd.GeoDataFrame:
    """Execute a SQL query and return results as GeoDataFrame.

    Args:
        sql: SQL query to execute.

    Returns:
        GeoDataFrame with query results.
    """
    logger.info("[sql_tool_utils] Executing SQL query")

    # Wrap geometry with ST_AsText for proper WKT output
    if re.search(r'\bSELECT\s+\*\s+FROM\b', sql, re.IGNORECASE):
        sql = re.sub(
            r'\bSELECT\s+\*\s+FROM\s+(\w+)',
            r'SELECT ST_AsText(geometry) as geometry, * EXCLUDE (geometry) FROM \1',
            sql,
            flags=re.IGNORECASE
        )

    logger.debug(f"[sql_tool_utils] SQL: {sql}")

    result = _md_service.execute_query(sql)
    df = result.fetchdf()

    if df.empty:
        logger.warning("[sql_tool_utils] Query returned no results")
        return gpd.GeoDataFrame()

    # Convert to GeoDataFrame if geometry column exists
    if "geometry" in df.columns:
        from shapely import wkt

        def parse_geometry(geom):
            """Parse geometry from WKT string."""
            if geom is None:
                return None
            if hasattr(geom, 'geom_type'):
                return geom
            if isinstance(geom, str):
                try:
                    return wkt.loads(geom)
                except Exception:
                    return None
            return None

        df["geometry"] = df["geometry"].apply(parse_geometry)
        df = df[df["geometry"].notna()]
        if df.empty:
            return gpd.GeoDataFrame()
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    else:
        gdf = gpd.GeoDataFrame(df)

    logger.info(f"[sql_tool_utils] Query returned {len(gdf)} features")
    return gdf
