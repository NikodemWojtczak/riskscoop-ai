import io
import json
import logging
from typing import Any, Dict, Union

import geopandas as gpd

logger = logging.getLogger(__name__)


class TypeConverter:
    """Service for converting geospatial data types.

    This service provides functionality for converting GeoDataFrames to various
    formats like GeoJSON. It handles data type conversions without managing
    file persistence (use LayerService for saving/loading layers).
    """

    def __init__(self) -> None:
        """Initialize the TypeConverter service."""
        logger.info("[type_converter] TypeConverter initialized")

    def convert_gdf_to_geojson(
        self, gdf: gpd.GeoDataFrame, output_format: str = "bytes"
    ) -> Union[bytes, Dict[str, Any]]:
        """Convert GeoDataFrame to GeoJSON format.

        This method converts a GeoDataFrame to GeoJSON format, handling multiple
        geometry columns by converting additional ones to WKT strings. Uses
        memory-efficient BytesIO processing for better performance.

        Args:
            gdf: The GeoDataFrame to convert to GeoJSON.
            output_format: The desired output format, either 'bytes' or 'dict'.
                Defaults to 'bytes'.

        Returns:
            GeoJSON data as bytes if output_format is 'bytes', or as a
            dictionary if output_format is 'dict'.

        Raises:
            ValueError: If output_format is not 'bytes' or 'dict'.

        Example:
            >>> converter = TypeConverter()
            >>> gdf = gpd.GeoDataFrame(...)
            >>> geojson_bytes = converter.convert_gdf_to_geojson(gdf, 'bytes')
            >>> geojson_dict = converter.convert_gdf_to_geojson(gdf, 'dict')

        Note:
            If the GeoDataFrame has multiple geometry columns, additional
            geometry columns will be converted to WKT strings and the
            columns will be renamed with a '_wkt' suffix.
        """
        if gdf.empty:
            logger.warning("[type_converter] Empty GeoDataFrame provided")
            empty_geojson = {"type": "FeatureCollection", "features": []}
            if output_format == "dict":
                return empty_geojson
            elif output_format == "bytes":
                return json.dumps(empty_geojson).encode("utf-8")
            else:
                raise ValueError("output_format must be 'bytes' or 'dict'")

        # Handle multiple geometry columns
        geometry_columns = gdf.select_dtypes(include=["geometry"]).columns
        if len(geometry_columns) > 1:
            # Create a copy to avoid SettingWithCopyWarning if gdf is a slice
            gdf = gdf.copy()
            for col in geometry_columns[1:]:
                # Convert additional geometry columns to WKT strings
                gdf[f"{col}_wkt"] = gdf[col].to_wkt()
                gdf = gdf.drop(columns=[col])

        # Use BytesIO instead of temporary file for better performance
        with io.BytesIO() as buffer:
            gdf.to_file(buffer, driver="GeoJSON")
            buffer.seek(0)
            geojson_bytes = buffer.read()

        if output_format == "dict":
            return json.loads(geojson_bytes)
        elif output_format == "bytes":
            return geojson_bytes
        else:
            raise ValueError("output_format must be 'bytes' or 'dict'")
