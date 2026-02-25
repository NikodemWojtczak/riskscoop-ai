import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd

from services.type_converter import TypeConverter

logger = logging.getLogger(__name__)


class LayerService:
    """Service for managing geospatial layers persistence and session state.

    This service handles saving GeoDataFrames as GeoJSON files with UUID filenames,
    loading layers back from disk, and managing layer references in session state.
    It acts as the central point for layer I/O operations in the application.
    """

    def __init__(self, output_dir: str = "output/geojson") -> None:
        """Initialize the LayerService.

        Args:
            output_dir: Directory path where GeoJSON files will be saved.
                Defaults to 'output/geojson'.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.type_converter = TypeConverter()
        logger.info(
            f"[layer_service] LayerService initialized with output dir: {self.output_dir}"
        )

    def save_layer(
        self,
        gdf: gpd.GeoDataFrame,
        session_state: Optional[Dict[str, Any]] = None,
        layer_name: Optional[str] = None,
    ) -> str:
        """Save a GeoDataFrame as a GeoJSON layer with UUID filename.

        This method converts a GeoDataFrame to GeoJSON format and saves it
        to the output directory with a unique UUID filename. Optionally updates
        the session state with the layer reference.

        Args:
            gdf: The GeoDataFrame to save as a layer.
            session_state: Optional session state dictionary to store the layer UUID.
                If provided along with layer_name, the UUID will be stored at
                session_state["layers"][layer_name].
            layer_name: Optional name for the layer in session state.
                Required if session_state is provided.

        Returns:
            The UUID string used as the filename (without extension).

        Example:
            >>> layer_service = LayerService()
            >>> gdf = gpd.GeoDataFrame(...)
            >>> session_state = {}
            >>> file_uuid = layer_service.save_layer(gdf, session_state, "my_layer")
            >>> print(session_state["layers"]["my_layer"])  # prints the UUID
        """
        file_uuid = str(uuid.uuid4())
        file_path = self.output_dir / f"{file_uuid}.geojson"

        geojson_bytes = self.type_converter.convert_gdf_to_geojson(
            gdf, output_format="bytes"
        )

        with open(file_path, "wb") as f:
            f.write(geojson_bytes)

        logger.info(f"[layer_service] Saved layer to: {file_path}")

        # Update session state if provided
        if session_state is not None and layer_name is not None:
            if "layers" not in session_state:
                session_state["layers"] = {}
            session_state["layers"][layer_name] = file_uuid
            logger.debug(
                f"[layer_service] Stored layer UUID in session_state['layers']['{layer_name}']"
            )

        return file_uuid

    def load_layer(self, file_uuid: str) -> gpd.GeoDataFrame:
        """Load a GeoJSON layer by its UUID.

        Args:
            file_uuid: The UUID of the layer file to load.

        Returns:
            GeoDataFrame loaded from the GeoJSON file.

        Raises:
            FileNotFoundError: If the layer file does not exist.

        Example:
            >>> layer_service = LayerService()
            >>> gdf = layer_service.load_layer("550e8400-e29b-41d4-a716-446655440000")
        """
        file_path = self.output_dir / f"{file_uuid}.geojson"

        if not file_path.exists():
            logger.error(f"[layer_service] Layer file not found: {file_path}")
            raise FileNotFoundError(f"Layer file not found: {file_path}")

        gdf = gpd.read_file(file_path)
        logger.info(f"[layer_service] Loaded layer from: {file_path}")
        return gdf

    def load_layer_by_name(
        self, session_state: Dict[str, Any], layer_name: str
    ) -> gpd.GeoDataFrame:
        """Load a layer by its name from session state.

        Args:
            session_state: Session state dictionary containing layer references.
            layer_name: Name of the layer to load.

        Returns:
            GeoDataFrame loaded from the GeoJSON file.

        Raises:
            KeyError: If the layer name is not found in session state.
            FileNotFoundError: If the layer file does not exist.

        Example:
            >>> layer_service = LayerService()
            >>> session_state = {"layers": {"my_layer": "550e8400-..."}}
            >>> gdf = layer_service.load_layer_by_name(session_state, "my_layer")
        """
        if "layers" not in session_state:
            raise KeyError("No layers found in session state")

        if layer_name not in session_state["layers"]:
            raise KeyError(f"Layer '{layer_name}' not found in session state")

        file_uuid = session_state["layers"][layer_name]
        return self.load_layer(file_uuid)

    def delete_layer(
        self,
        file_uuid: str,
        session_state: Optional[Dict[str, Any]] = None,
        layer_name: Optional[str] = None,
    ) -> bool:
        """Delete a layer file and optionally remove from session state.

        Args:
            file_uuid: The UUID of the layer file to delete.
            session_state: Optional session state to remove layer reference from.
            layer_name: Optional layer name to remove from session state.

        Returns:
            True if the file was deleted, False if it didn't exist.

        Example:
            >>> layer_service = LayerService()
            >>> layer_service.delete_layer("550e8400-...", session_state, "my_layer")
        """
        file_path = self.output_dir / f"{file_uuid}.geojson"

        if file_path.exists():
            file_path.unlink()
            logger.info(f"[layer_service] Deleted layer: {file_path}")
            deleted = True
        else:
            logger.warning(f"[layer_service] Layer file not found for deletion: {file_path}")
            deleted = False

        # Remove from session state if provided
        if session_state is not None and layer_name is not None:
            if "layers" in session_state and layer_name in session_state["layers"]:
                del session_state["layers"][layer_name]
                logger.debug(
                    f"[layer_service] Removed layer '{layer_name}' from session state"
                )

        return deleted

    def list_layers(self, session_state: Dict[str, Any]) -> Dict[str, str]:
        """List all layers in session state.

        Args:
            session_state: Session state dictionary containing layer references.

        Returns:
            Dictionary mapping layer names to their UUIDs.

        Example:
            >>> layer_service = LayerService()
            >>> layers = layer_service.list_layers(session_state)
            >>> print(layers)  # {"my_layer": "550e8400-...", "other_layer": "..."}
        """
        return session_state.get("layers", {}).copy()

    def layer_exists(self, file_uuid: str) -> bool:
        """Check if a layer file exists.

        Args:
            file_uuid: The UUID of the layer file to check.

        Returns:
            True if the layer file exists, False otherwise.
        """
        file_path = self.output_dir / f"{file_uuid}.geojson"
        return file_path.exists()

    def get_layer_path(self, file_uuid: str) -> Path:
        """Get the full file path for a layer UUID.

        Args:
            file_uuid: The UUID of the layer.

        Returns:
            Path object for the layer file.
        """
        return self.output_dir / f"{file_uuid}.geojson"
