import logging
import os
import threading
from typing import Optional

import duckdb
from dotenv import load_dotenv

load_dotenv(override=False)

logger = logging.getLogger(__name__)


class DatabaseEngine:
    """
    Centralized database engine for MotherDuck connections.

    This class implements a singleton pattern to ensure only one instance
    of the database engine exists across the entire application. It uses
    thread-local storage to provide each thread with its own connection,
    making it safe for parallel execution.

    Usage:
        # Get the singleton instance
        db_engine = DatabaseEngine.get_instance()

        # Execute queries (thread-safe)
        result = db_engine.execute_query("SELECT * FROM spatial_features")
    """

    _instance: Optional["DatabaseEngine"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, "_initialized"):
            return

        logger.info(
            "[database_engine] Initializing DatabaseEngine singleton (lazy mode - no connection yet)"
        )

        try:
            # Install global extensions (only needed once) with error handling
            try:
                duckdb.sql("INSTALL motherduck;")
                logger.debug(
                    "[database_engine] Successfully installed motherduck extension"
                )
            except Exception as e:
                if "already installed" in str(e).lower():
                    logger.debug(
                        "[database_engine] motherduck extension already installed"
                    )
                else:
                    logger.warning(
                        f"[database_engine] Failed to install motherduck extension: {e}"
                    )

            try:
                duckdb.sql("LOAD motherduck;")
                logger.debug(
                    "[database_engine] Successfully loaded motherduck extension"
                )
            except Exception as e:
                if "already loaded" in str(e).lower():
                    logger.debug(
                        "[database_engine] motherduck extension already loaded"
                    )
                else:
                    logger.warning(
                        f"[database_engine] Failed to load motherduck extension: {e}"
                    )

            motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
            if not motherduck_token:
                raise ValueError("MOTHERDUCK_TOKEN environment variable is required")

            self.db_path = "md:geoforge_db"
            self.config = {"motherduck_token": motherduck_token}

            # Thread-local storage for connections
            self._thread_local = threading.local()

            # Flag to track if tables have been setup
            self._tables_setup = False
            self._tables_setup_lock = threading.Lock()

            self._initialized = True
            logger.info(
                "[database_engine] DatabaseEngine initialized (will connect on first query)"
            )

        except Exception as e:
            logger.error(
                f"[database_engine] Error initializing DatabaseEngine: {str(e)}",
                exc_info=True,
            )
            raise

    def _setup_tables(self):
        """Setup required database tables synchronously."""
        # Create a temporary connection for setup
        conn = duckdb.connect(self.db_path, config=self.config)
        try:
            # Setup extensions with error handling
            self._setup_extensions(conn)

            # Create table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spatial_features (
                    layer_id TEXT,
                    feature_id TEXT,
                    geometry GEOMETRY,
                );
            """
            )
        finally:
            conn.close()

    def _setup_extensions(self, conn):
        """Setup required extensions with error handling."""
        extensions = [
            ("spatial", "INSTALL spatial;", "LOAD spatial;"),
            ("httpfs", "INSTALL httpfs;", "LOAD httpfs;"),
        ]

        for ext_name, install_cmd, load_cmd in extensions:
            try:
                # Try to install the extension
                conn.execute(install_cmd)
                logger.debug(
                    f"[database_engine] Successfully installed {ext_name} extension"
                )
            except Exception as e:
                # If installation fails, check if it's already installed
                if (
                    "already installed" in str(e).lower()
                    or "existing extension" in str(e).lower()
                ):
                    logger.debug(
                        f"[database_engine] {ext_name} extension already installed"
                    )
                else:
                    logger.warning(
                        f"[database_engine] Failed to install {ext_name} extension: {e}"
                    )
                    # Continue anyway as the extension might already be available

            try:
                # Try to load the extension
                conn.execute(load_cmd)
                logger.debug(
                    f"[database_engine] Successfully loaded {ext_name} extension"
                )
            except Exception as e:
                if "already loaded" in str(e).lower():
                    logger.debug(
                        f"[database_engine] {ext_name} extension already loaded"
                    )
                else:
                    logger.warning(
                        f"[database_engine] Failed to load {ext_name} extension: {e}"
                    )
                    # Continue anyway as the extension might already be available

    @staticmethod
    def _setup_extensions_for_connection(conn):
        """Setup extensions for a single connection with error handling."""
        extensions = [
            ("spatial", "INSTALL spatial;", "LOAD spatial;"),
            ("httpfs", "INSTALL httpfs;", "LOAD httpfs;"),
        ]

        for ext_name, install_cmd, load_cmd in extensions:
            try:
                # Try to install the extension
                conn.execute(install_cmd)
            except Exception as e:
                # If installation fails, check if it's already installed
                if (
                    "already installed" in str(e).lower()
                    or "existing extension" in str(e).lower()
                ):
                    pass  # Extension already installed, continue
                else:
                    logger.warning(
                        f"[database_engine] Failed to install {ext_name} extension: {e}"
                    )
                    # Continue anyway as the extension might already be available

            try:
                # Try to load the extension
                conn.execute(load_cmd)
            except Exception as e:
                if "already loaded" in str(e).lower():
                    pass  # Extension already loaded, continue
                else:
                    logger.warning(
                        f"[database_engine] Failed to load {ext_name} extension: {e}"
                    )
                    # Continue anyway as the extension might already be available

    def _get_thread_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create a thread-local connection.

        Each thread gets its own connection that is reused for all queries
        within that thread. This ensures thread safety without locks.

        Returns:
            duckdb.DuckDBPyConnection: Thread-local database connection
        """
        if not hasattr(self._thread_local, "connection"):
            logger.debug(
                f"[database_engine] Creating new connection for thread {threading.current_thread().name}"
            )

            # Setup tables on first connection (thread-safe)
            if not self._tables_setup:
                with self._tables_setup_lock:
                    if not self._tables_setup:
                        logger.info(
                            "[database_engine] First connection - setting up tables"
                        )
                        self._setup_tables()
                        self._tables_setup = True

            conn = duckdb.connect(self.db_path, config=self.config)

            # Setup extensions for this connection
            self._setup_extensions_for_connection(conn)

            self._thread_local.connection = conn

        return self._thread_local.connection

    def execute_query(
        self, query: str, timeout_seconds: int = 120
    ) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query using thread-local connection.

        Args:
            query (str): The SQL query to execute.
            timeout_seconds (int): Maximum time in seconds to wait for query execution.

        Returns:
            duckdb.DuckDBPyRelation: The relation object resulting from the query execution.

        Raises:
            Exception: If the query execution fails.

        Note:
            Timeout handling is minimal here since DuckDB doesn't support query cancellation.
            For long-running queries, consider implementing at a higher level.
        """
        logger.debug(
            f"[database_engine] Executing query with {timeout_seconds}s timeout: %s",
            query,
        )

        try:
            conn = self._get_thread_connection()
            result = conn.execute(query)

            logger.debug("[database_engine] Query execution completed within timeout.")
            return result

        except Exception as e:
            logger.error(f"[database_engine] Query execution failed: {str(e)}")
            logger.debug(f"[database_engine] Failed query: {query}")
            raise e

    @classmethod
    def get_instance(cls) -> "DatabaseEngine":
        """Get the singleton instance of DatabaseEngine."""
        return cls()


import logging

import duckdb
import geopandas as gpd
from dotenv import load_dotenv
from shapely import wkt


load_dotenv(override=False)


# Configure logging
logger = logging.getLogger(__name__)


class MDService:
    """
    MotherDuck Database Service - Lightweight wrapper around DatabaseEngine.

    This service provides a convenient interface for MotherDuck operations
    while using the centralized DatabaseEngine for connection management.

    The MDService maintains backward compatibility with existing code while
    leveraging the singleton DatabaseEngine to avoid multiple connection instances.
    """

    def __init__(self):
        """Initialize the MotherDuck Database Service.
        Note:
            This constructor now uses the centralized DatabaseEngine singleton
            instead of creating its own connections.
        """
        logger.info(
            "[md_service] Initializing MDService with centralized DatabaseEngine"
        )

        # Get the singleton database engine instance
        self.db_engine = DatabaseEngine.get_instance()

        logger.info("[md_service] Successfully initialized MDService")

    def execute_query(
        self, query: str, timeout_seconds: int = 120
    ) -> duckdb.DuckDBPyRelation:
        """Synchronous wrapper for execute_query to maintain backward compatibility.

        Args:
            query (str): The SQL query to execute.
            timeout_seconds (int): The maximum time in seconds to wait for query execution.

        Returns:
            duckdb.DuckDBPyRelation: The relation object resulting from the query execution.
        """
        return self.db_engine.execute_query(query, timeout_seconds)

    def delete_layer_from_motherduck(self, layer_id: str):
        """Delete a layer from MotherDuck spatial_features table.

        Args:
            layer_id (str): The unique identifier of the layer to delete.

        Raises:
            ValueError: If the deletion fails.
        """
        try:
            logger.info(
                f"[md_service] Deleting layer {layer_id} from MotherDuck spatial_features table"
            )
            query = f"""
            DELETE FROM spatial_features
            WHERE layer_id = '{layer_id}'
            """
            self.execute_query(query)
            logger.info(
                f"[md_service] Successfully deleted layer {layer_id} from MotherDuck"
            )
        except Exception as e:
            logger.error(
                f"[md_service] Failed to delete layer {layer_id} from MotherDuck: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to delete layer from MotherDuck: {str(e)}")

    def get_layer_from_motherduck(self, layer_id: str) -> gpd.GeoDataFrame:
        """Retrieve layer from the database.

        Args:
            layer_id (str): The unique identifier of the layer.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing all geometries for the layer.

        Raises:
            ValueError: If the layer is not found or retrieval fails.
        """
        try:
            logger.info(f"[md_service] Retrieving geometries for layer {layer_id}")
            query = f"""
                    SELECT ST_AsText(geometry) as geometry, feature_id
                    FROM spatial_features
                    WHERE layer_id = '{layer_id}';
                """
            result = self.execute_query(query)
            df = result.fetchdf()

            if df.empty:
                logger.warning(f"[md_service] No geometries found for layer {layer_id}")
                return gpd.GeoDataFrame()

            logger.debug(
                f"[md_service] Converting {len(df)} geometries from WKT to geometry objects for layer {layer_id}."
            )
            df["geometry"] = df["geometry"].apply(
                lambda x: wkt.loads(x) if isinstance(x, str) else None
            )

            # Check if any geometries are None after conversion
            if df["geometry"].isnull().all():
                logger.warning(
                    f"[md_service] Failed to convert WKT to geometry objects for layer {layer_id}."
                )

            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            logger.info(
                f"[md_service] Successfully retrieved {len(gdf)} geometries for layer {layer_id}."
            )
            return gdf

        except Exception as e:
            logger.error(
                f"[md_service] Failed to retrieve geometries for layer {layer_id}: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to retrieve geometries from MotherDuck: {str(e)}")

    def get_layer_extent(self, layer_id: str) -> dict:
        """Get the extent of a layer using SQL.

        Args:
            layer_id (str): The unique identifier of the layer.

        Returns:
            dict: A dictionary containing the layer extent with keys:
                minx: Minimum X coordinate
                miny: Minimum Y coordinate
                maxx: Maximum X coordinate
                maxy: Maximum Y coordinate

        Raises:
            ValueError: If the layer is not found or has no geometry.
        """
        try:
            logger.debug(f"[md_service] Getting extent for layer {layer_id}")
            query = f"""
            SELECT
                MIN(ST_XMin(geometry)) AS minx,
                MIN(ST_YMin(geometry)) AS miny,
                MAX(ST_XMax(geometry)) AS maxx,
                MAX(ST_YMax(geometry)) AS maxy
            FROM spatial_features
            WHERE layer_id = '{layer_id}'
            """

            result = self.execute_query(query)
            result_df = result.fetchdf()

            if result_df.empty or result_df.iloc[0][0] is None:
                logger.warning(
                    f"[md_service] Layer {layer_id} not found or has no geometry"
                )
                raise ValueError(f"Layer {layer_id} not found or has no geometry")

            bbox_dict = {
                "minx": result_df.iloc[0][0],
                "miny": result_df.iloc[0][1],
                "maxx": result_df.iloc[0][2],
                "maxy": result_df.iloc[0][3],
            }

            logger.debug(
                f"[md_service] Successfully retrieved extent for layer {layer_id}: {bbox_dict}"
            )
            return bbox_dict

        except ValueError as e:
            # Re-raise ValueError with same message
            logger.error(f"[md_service] {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"[md_service] Failed to get extent for layer {layer_id}: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to get layer extent: {str(e)}")

    def get_geometry_types(self, db_name, xmin, ymin, xmax, ymax):
        """Detect existing geometry types in the dataset within specified bounds.

        Args:
            db_name (str): The name of the database.
            xmin (float): Minimum X coordinate of the bounding box.
            ymin (float): Minimum Y coordinate of the bounding box.
            xmax (float): Maximum X coordinate of the bounding box.
            ymax (float): Maximum Y coordinate of the bounding box.

        Returns:
            set: A set of geometry types found in the dataset.

        Raises:
            ValueError: If the geometry type detection fails.
        """
        try:
            logger.debug(
                f"[md_service] Detecting geometry types for {db_name} in bounds ({xmin}, {ymin}, {xmax}, {ymax})"
            )
            # Detect existing geometry types in the dataset
            detect_types_query = f"""
            SELECT DISTINCT ST_GeometryType(geometry)
            FROM geoforge_db.main.{db_name}
            WHERE bbox.xmin > {xmin}
            AND bbox.ymin > {ymin}
            AND bbox.xmax < {xmax}
            AND bbox.ymax < {ymax}
            """

            result = self.execute_query(detect_types_query)
            existing_geom_types = {row[0] for row in result.fetchall()}

            if not existing_geom_types:
                logger.warning(
                    f"[md_service] No geometry types found for {db_name} in the specified bounds"
                )
            else:
                logger.debug(
                    f"[md_service] Found geometry types for {db_name}: {existing_geom_types}"
                )

            return existing_geom_types

        except Exception as e:
            logger.error(
                f"[md_service] Failed to detect geometry types for {db_name}: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to detect geometry types: {str(e)}")

    def get_polygon_from_motherduck(self, layer_id: str) -> gpd.GeoDataFrame:
        """Retrieve polygon geometries from the database.

        Args:
            layer_id (str): The unique identifier of the layer.

        Returns:
            gpd.GeoDataFrame: A GeoDataFrame containing all polygon geometries for the layer.

        Raises:
            ValueError: If the layer is not found or retrieval fails.
        """
        try:
            logger.info(
                f"[md_service] Retrieving polygon geometries for layer {layer_id}"
            )
            query = f"""
                    SELECT ST_AsText(geometry) as geometry, feature_id
                    FROM spatial_features
                    WHERE layer_id = '{layer_id}';
                """
            result = self.execute_query(query)
            df = result.fetchdf()

            if df.empty:
                logger.warning(f"[md_service] No geometries found for layer {layer_id}")
                return gpd.GeoDataFrame()

            logger.debug(
                f"[md_service] Converting {len(df)} geometries from WKT to geometry objects for layer {layer_id}"
            )
            df["geometry"] = df["geometry"].apply(
                lambda x: wkt.loads(x) if isinstance(x, str) else None
            )

            # Check if any geometries are None after conversion
            if df["geometry"].isnull().all():
                logger.warning(
                    f"[md_service] Failed to convert WKT to geometry objects for layer {layer_id}"
                )

            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            logger.info(
                f"[md_service] Successfully retrieved {len(gdf)} geometries for layer {layer_id}"
            )
            return gdf

        except Exception as e:
            logger.error(
                f"[md_service] Failed to retrieve polygon geometries for layer {layer_id}: {str(e)}",
                exc_info=True,
            )
            raise ValueError(f"Failed to retrieve polygons from MotherDuck: {str(e)}")
