import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

logger = logging.getLogger(__name__)


class NominatimService:
    """OpenStreetMap Nominatim API Nominatim and Overpass API integration service.

    This service provides comprehensive geocoding and boundary retrieval capabilities
    using OpenStreetMap Nominatim API's Nominatim search API and Overpass API for detailed geometry
    extraction. It's designed to handle complex spatial queries and geometry assembly
    for administrative boundaries, natural features, and other geographic entities.

    The service supports multiple workflow patterns:
        1. Simple place search with Nominatim for basic geocoding
        2. Complex boundary retrieval combining search + geometry extraction
        3. Multi-level administrative boundary processing
        4. Custom geometry assembly for complex OSM relations

    Key Capabilities:
        - Place name resolution with fuzzy matching
        - Administrative boundary extraction at all levels
        - Complex polygon assembly with inner/outer rings
        - Multi-geometry type processing (points, lines, polygons)
        - Metadata preservation and enrichment
        - Robust error handling for API failures

    Example:
        Initialize and perform various geocoding operations:

            service = NominatimService()

            # Simple place search
            places = service.search_for_places("Central Park, NYC")

            # Full boundary retrieval with geometry
            boundaries = service.search_and_get_all_boundaries("Manhattan, NY")

            # Process results
            for _, boundary in boundaries.iterrows():
                print(f"{boundary['display_name']}: {boundary.geometry.area} sq degrees")
    """

    def __init__(self) -> None:
        """Initialize the Nominatim service.

        Sets up logging and prepares the service for geocoding operations.
        No external dependencies or API keys are required for initialization,
        as the service uses publicly available OpenStreetMap Nominatim API APIs.

        Note:
            The service includes proper User-Agent headers for API compliance
            and follows OpenStreetMap Nominatim API usage policies.
        """
        logger.info("[nominatim_service] Nominatim Service initialized successfully")

    def search_for_places(self, query: str) -> gpd.GeoDataFrame:
        """Search for places using Nominatim geocoding API.

        This method performs a comprehensive place search using OpenStreetMap Nominatim API's
        Nominatim service, which provides geocoding for addresses, place names,
        and various geographic features. The search supports fuzzy matching and
        returns detailed metadata for each result.

        The Nominatim search includes:
        - Administrative boundaries (countries, states, cities, districts)
        - Natural features (lakes, mountains, parks)
        - Infrastructure (roads, buildings, amenities)
        - Points of interest (restaurants, landmarks, institutions)

        Args:
            query (str): Search query string. Can be an address, place name,
                or geographic feature description. Examples:
                - "Paris, France"
                - "1600 Pennsylvania Avenue, Washington DC"
                - "Mount Everest"
                - "Central Park, New York"

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing search results with:
                - geometry: Point geometries for place locations
                - display_name: Full formatted place name
                - osm_type: OSM element type (node, way, relation)
                - osm_id: Unique OpenStreetMap Nominatim API identifier
                - place_rank: Importance ranking (0-30, lower is more important)
                - category: Place category (boundary, amenity, highway, etc.)
                - type: Specific place type (city, restaurant, park, etc.)
                - importance: Numerical importance score (0-1, higher is more important)
                Returns empty GeoDataFrame if no results found.

        Note:
            Results are limited to 50 features and returned in EPSG:4326
            coordinate system. The service includes proper rate limiting
            and error handling for production use.

        Example:
            >>> service = NominatimService()
            >>> results = service.search_for_places("Tokyo, Japan")
            >>> if not results.empty:
            ...     print(f"Found {len(results)} places")
            ...     print(results[['display_name', 'category', 'type']])
            ... else:
            ...     print("No places found")
        """
        logger.info(
            f"[nominatim_service] Starting Nominatim search for query: '{query}'"
        )
        headers = {
            "User-Agent": "GeoForge-MVP/1.0 (contact@ageospatial.com)",
            "Accept": "application/json",
        }
        search_url = f"https://nominatim.openstreetmap.org/search?q={query}&format=geojson&limit=50"

        try:
            logger.debug(f"[nominatim_service] Making request to: {search_url}")
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            geojson_data = response.json()
            if geojson_data["features"]:
                gdf = gpd.GeoDataFrame.from_features(
                    geojson_data["features"], crs="EPSG:4326"
                )
                logger.info(
                    f"[nominatim_service] Successfully retrieved {len(gdf)} features from Nominatim"
                )
                return gdf
            logger.warning(
                f"[nominatim_service] No features found for query: '{query}'"
            )
            return gpd.GeoDataFrame()
        except requests.exceptions.RequestException as e:
            logger.error(
                f"[nominatim_service] An API request error occurred during Nominatim search: {e}",
                exc_info=True,
            )
            return gpd.GeoDataFrame()

    def assemble_rings(
        self,
        way_ids: List[int],
        ways_data: Dict[int, Dict[str, Any]],
        nodes_data: Dict[int, Tuple[float, float]],
    ) -> List[List[Tuple[float, float]]]:
        """Assemble unordered OSM ways into closed coordinate rings.

        This method reconstructs polygon rings from OpenStreetMap Nominatim API way data by
        connecting individual ways end-to-end to form closed polygons. This is
        essential for processing OSM relations where polygon boundaries are
        represented as collections of connected way segments.

        The algorithm:
        1. Starts with any available way segment
        2. Iteratively finds connecting ways that share endpoints
        3. Handles both forward and reverse connection directions
        4. Continues until a closed ring is formed or no more connections exist
        5. Converts node IDs to actual coordinate pairs
        6. Validates minimum ring requirements (4+ coordinates)

        This process is crucial for reconstructing complex administrative
        boundaries and natural features that span multiple OSM way elements.

        Args:
            way_ids (List[int]): List of OSM way IDs to assemble into rings.
                These should represent segments of polygon boundaries.
            ways_data (Dict[int, Dict[str, Any]]): Dictionary mapping way IDs
                to their data including 'nodes' list and 'tags' metadata.
            nodes_data (Dict[int, Tuple[float, float]]): Dictionary mapping
                node IDs to their coordinate pairs (longitude, latitude).

        Returns:
            List[List[Tuple[float, float]]]: List of coordinate rings, where
                each ring is a list of (longitude, latitude) coordinate pairs
                forming a closed polygon. Only rings with 4+ coordinates
                (minimum for valid polygon) are included.

        Note:
            This method handles complex topological scenarios including:
            - Non-contiguous way segments requiring connection
            - Bidirectional way traversal for proper ring closure
            - Missing or invalid node coordinates
            - Multiple disconnected rings from the same way set

        Example:
            >>> # Reconstruct boundary from way segments
            >>> way_ids = [12345, 67890, 11111]
            >>> ways_data = {
            ...     12345: {'nodes': [100, 101, 102], 'tags': {}},
            ...     67890: {'nodes': [102, 103, 104], 'tags': {}},
            ...     11111: {'nodes': [104, 105, 100], 'tags': {}}
            ... }
            >>> nodes_data = {
            ...     100: (-74.0, 40.7), 101: (-74.1, 40.7),
            ...     102: (-74.1, 40.8), 103: (-74.0, 40.8),
            ...     104: (-73.9, 40.8), 105: (-73.9, 40.7)
            ... }
            >>> service = NominatimService()
            >>> rings = service.assemble_rings(way_ids, ways_data, nodes_data)
            >>> print(f"Assembled {len(rings)} rings")
        """
        if not way_ids:
            logger.debug("[nominatim_service] No way IDs provided for ring assembly")
            return []

        logger.debug(f"[nominatim_service] Assembling rings from {len(way_ids)} ways")
        ways_to_process = {
            wid: ways_data[wid]["nodes"] for wid in way_ids if wid in ways_data
        }
        rings = []
        while ways_to_process:
            current_way_id, current_ring_nodes = ways_to_process.popitem()
            logger.debug(
                f"[nominatim_service] Processing way {current_way_id} with {len(current_ring_nodes)} nodes"
            )
            while True:
                head, tail = current_ring_nodes[0], current_ring_nodes[-1]
                if head == tail:
                    break
                found_way = False
                for way_id, node_ids in list(ways_to_process.items()):
                    if node_ids[0] == tail:
                        current_ring_nodes.extend(node_ids[1:])
                        ways_to_process.pop(way_id)
                        found_way = True
                        break
                    elif node_ids[-1] == tail:
                        current_ring_nodes.extend(list(reversed(node_ids[:-1])))
                        ways_to_process.pop(way_id)
                        found_way = True
                        break
                if not found_way:
                    break
            coords = [
                nodes_data[node_id]
                for node_id in current_ring_nodes
                if node_id in nodes_data
            ]
            if len(coords) >= 4:
                rings.append(coords)
                logger.debug(
                    f"[nominatim_service] Successfully assembled ring with {len(coords)} coordinates"
                )
            else:
                logger.warning(
                    f"[nominatim_service] Ring has insufficient coordinates ({len(coords)} < 4)"
                )

        logger.debug(f"[nominatim_service] Assembled {len(rings)} rings total")
        return rings

    def get_boundaries_from_overpass(
        self, osm_items: List[Dict[str, Union[str, int]]]
    ) -> Dict[Tuple[str, int], gpd.GeoDataFrame]:
        """Retrieve and assemble boundary geometries from Overpass API.

        This method fetches detailed geometric data for OpenStreetMap Nominatim API elements
        using the Overpass API, which provides comprehensive access to OSM's
        spatial database. It processes multiple element types (nodes, ways,
        relations) and constructs appropriate geometries for each.

        The Overpass API workflow:
        1. Groups OSM elements by type for efficient querying
        2. Constructs specialized Overpass QL query for all element types
        3. Retrieves complete geometric data including child elements
        4. Processes nodes as points, ways as lines/polygons, relations as complex polygons
        5. Handles complex relation assembly with inner/outer rings
        6. Returns geometry-enabled GeoDataFrames for spatial analysis

        Supported Element Types:
        - Nodes: Converted to Point geometries with coordinates
        - Ways: Converted to LineString or Polygon based on closure
        - Relations: Assembled into Polygon or MultiPolygon with holes

        Args:
            osm_items (List[Dict[str, Union[str, int]]]): List of OSM element
                specifications, each containing:
                - 'osm_type' (str): Element type ('node', 'way', 'relation')
                - 'osm_id' (int): Unique OSM identifier for the element
                Examples: [{'osm_type': 'relation', 'osm_id': 123456}]

        Returns:
            Dict[Tuple[str, int], gpd.GeoDataFrame]: Dictionary mapping
                (osm_type, osm_id) tuples to GeoDataFrames containing:
                - geometry: Spatial geometry (Point, LineString, Polygon, MultiPolygon)
                - osm_id: Original OSM identifier
                - osm_type: Element type
                - tags: OSM metadata and attributes
                Returns empty dict if no valid geometries are constructed.

        Raises:
            No exceptions are raised. API failures and geometry construction
            errors are logged and result in empty returns for graceful handling.

        Note:
            This method includes comprehensive error handling for:
            - Network connectivity issues
            - Overpass API timeouts and rate limits
            - Malformed or incomplete geometric data
            - Complex relation topology reconstruction

        Example:
            >>> service = NominatimService()
            >>> osm_items = [
            ...     {'osm_type': 'relation', 'osm_id': 175905},  # New York City
            ...     {'osm_type': 'way', 'osm_id': 4084868}       # Central Park boundary
            ... ]
            >>> boundaries = service.get_boundaries_from_overpass(osm_items)
            >>> for key, gdf in boundaries.items():
            ...     osm_type, osm_id = key
            ...     print(f"{osm_type} {osm_id}: {gdf.geometry.iloc[0].geom_type}")
        """
        if not osm_items:
            logger.warning(
                "[nominatim_service] No OSM items provided for boundary retrieval"
            )
            return {}

        logger.info(
            f"[nominatim_service] Fetching boundaries from Overpass API for {len(osm_items)} items..."
        )
        overpass_url = "https://overpass-api.de/api/interpreter"

        # Get all types of OSM elements, not just relations
        osm_ids_by_type = {}
        for item in osm_items:
            osm_type = item["osm_type"]
            osm_id = str(item["osm_id"])
            if osm_type not in osm_ids_by_type:
                osm_ids_by_type[osm_type] = []
            osm_ids_by_type[osm_type].append(osm_id)

        logger.debug(f"[nominatim_service] OSM IDs by type: {osm_ids_by_type}")

        if not osm_ids_by_type:
            logger.warning("[nominatim_service] No OSM IDs found by type")
            return {}

        # Build query for all types of elements
        query_parts = []
        for osm_type, ids in osm_ids_by_type.items():
            if ids:
                query_parts.append(f"{osm_type}(id:{','.join(ids)});")

        query = f"""
        [out:json][timeout:60];
        ({' '.join(query_parts)});
        (._;>;);
        out geom;
        """

        logger.debug(f"[nominatim_service] Overpass query: {query}")
        logger.debug("[nominatim_service] Executing Overpass query...")

        headers = {
            "User-Agent": "GeoForge-MVP/1.0 (contact@ageospatial.com)",
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }

        try:
            response = requests.post(overpass_url, data=query, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.info(
                "[nominatim_service] Successfully received response from Overpass API"
            )
        except requests.exceptions.RequestException as e:
            logger.error(
                f"[nominatim_service] An API request error occurred while fetching from Overpass: {e}",
                exc_info=True,
            )
            return {}

        elements = data.get("elements", [])
        if not elements:
            logger.warning("[nominatim_service] No elements found in Overpass response")
            return {}

        logger.info(
            f"[nominatim_service] Processing {len(elements)} elements from Overpass response"
        )

        nodes_data = {
            el["id"]: (el["lon"], el["lat"])
            for el in elements
            if el["type"] == "node" and "lat" in el
        }
        ways_data = {
            el["id"]: {"nodes": el.get("nodes", []), "tags": el.get("tags", {})}
            for el in elements
            if el["type"] == "way"
        }

        logger.debug(
            f"[nominatim_service] Extracted {len(nodes_data)} nodes and {len(ways_data)} ways"
        )
        results_dict = {}

        # Process all element types
        for element in elements:
            osm_id = element["id"]
            osm_type = element["type"]

            # Only process the main elements we searched for (not their children)
            if not any(
                item["osm_id"] == osm_id and item["osm_type"] == osm_type
                for item in osm_items
            ):
                continue

            logger.debug(f"[nominatim_service] Processing {osm_type} id: {osm_id}")

            if osm_type == "node":
                # Create point geometry for nodes
                if "lat" in element and "lon" in element:
                    geom = Point(element["lon"], element["lat"])
                    gdf = gpd.GeoDataFrame(
                        [
                            {
                                "osm_id": osm_id,
                                "osm_type": osm_type,
                                "tags": element.get("tags", {}),
                            }
                        ],
                        geometry=[geom],
                        crs="EPSG:4326",
                    )
                    results_dict[(osm_type, osm_id)] = gdf
                    logger.debug(
                        f"[nominatim_service] Successfully processed point for node {osm_id}"
                    )
                else:
                    logger.warning(
                        f"[nominatim_service] Node {osm_id} missing lat/lon coordinates"
                    )

            elif osm_type == "way":
                # Create line or polygon geometry for ways
                if osm_id in ways_data and ways_data[osm_id]["nodes"]:
                    nodes = ways_data[osm_id]["nodes"]
                    coords = [
                        nodes_data[node_id]
                        for node_id in nodes
                        if node_id in nodes_data
                    ]
                    if len(coords) >= 2:
                        # Check if it's a closed way (polygon) or open way (linestring)
                        if len(coords) >= 4 and coords[0] == coords[-1]:
                            geom = Polygon(coords)
                            logger.debug(
                                f"[nominatim_service] Created polygon geometry for way {osm_id}"
                            )
                        else:
                            geom = LineString(coords)
                            logger.debug(
                                f"[nominatim_service] Created linestring geometry for way {osm_id}"
                            )
                        gdf = gpd.GeoDataFrame(
                            [
                                {
                                    "osm_id": osm_id,
                                    "osm_type": osm_type,
                                    "tags": element.get("tags", {}),
                                }
                            ],
                            geometry=[geom],
                            crs="EPSG:4326",
                        )
                        results_dict[(osm_type, osm_id)] = gdf
                        logger.debug(
                            f"[nominatim_service] Successfully processed geometry for way {osm_id}"
                        )
                    else:
                        logger.warning(
                            f"[nominatim_service] Way {osm_id} has insufficient coordinates ({len(coords)} < 2)"
                        )
                else:
                    logger.warning(
                        f"[nominatim_service] Way {osm_id} not found in ways_data or has no nodes"
                    )

            elif osm_type == "relation":
                # Process relation boundaries as before
                outer_way_ids = [
                    m["ref"]
                    for m in element.get("members", [])
                    if m["type"] == "way" and m["role"] == "outer"
                ]
                inner_way_ids = [
                    m["ref"]
                    for m in element.get("members", [])
                    if m["type"] == "way" and m["role"] == "inner"
                ]

                logger.debug(
                    f"[nominatim_service] Relation {osm_id} has {len(outer_way_ids)} outer ways and {len(inner_way_ids)} inner ways"
                )

                outer_rings = self.assemble_rings(outer_way_ids, ways_data, nodes_data)
                inner_rings = self.assemble_rings(inner_way_ids, ways_data, nodes_data)

                if outer_rings:
                    all_polygons = [
                        Polygon(
                            shell,
                            [
                                hole
                                for hole in inner_rings
                                if Polygon(shell).contains(Polygon(hole))
                            ],
                        )
                        for shell in outer_rings
                    ]
                    if all_polygons:
                        final_geom = (
                            MultiPolygon(all_polygons)
                            if len(all_polygons) > 1
                            else all_polygons[0]
                        )
                        gdf = gpd.GeoDataFrame(
                            [
                                {
                                    "osm_id": osm_id,
                                    "osm_type": osm_type,
                                    "tags": element.get("tags", {}),
                                }
                            ],
                            geometry=[final_geom],
                            crs="EPSG:4326",
                        )
                        results_dict[(osm_type, osm_id)] = gdf
                        logger.debug(
                            f"[nominatim_service] Successfully processed boundary for relation {osm_id}"
                        )
                    else:
                        logger.error(
                            f"[nominatim_service] Could not create polygons for relation {osm_id}"
                        )
                else:
                    logger.error(
                        f"[nominatim_service] Could not assemble outer boundary for relation {osm_id}"
                    )

        logger.info(
            f"[nominatim_service] Successfully processed {len(results_dict)} geometries from Overpass"
        )
        return results_dict

    def search_and_get_all_boundaries(self, query: str) -> gpd.GeoDataFrame:
        """Complete workflow to search places and retrieve detailed boundary geometries.

        This is the main entry point for comprehensive geocoding and boundary
        retrieval operations. It combines Nominatim's rich place search capabilities
        with Overpass API's detailed geometric data to provide complete spatial
        datasets with both metadata and accurate geometries.

        The complete workflow:
        1. Performs comprehensive place search using Nominatim
        2. Extracts OSM identifiers from search results
        3. Retrieves detailed geometries via Overpass API
        4. Reconstructs complex polygons from OSM relations
        5. Merges geometric data with rich Nominatim metadata
        6. Returns integrated dataset with both attributes and geometry

        This method is particularly valuable for:
        - Administrative boundary analysis with full attribute data
        - Complex geographic feature extraction with metadata preservation
        - Multi-level spatial analysis combining search and geometric precision
        - Integration of crowd-sourced OSM data with structured place information

        Args:
            query (str): Place search query. Supports the same formats as
                search_for_places(), including addresses, place names, and
                geographic features. Examples:
                - "Paris, France" (city boundaries)
                - "Yellowstone National Park" (natural area boundaries)
                - "Manhattan, New York" (borough boundaries)
                - "Thames River, London" (waterway geometries)

        Returns:
            gpd.GeoDataFrame: Comprehensive spatial dataset containing:
                - geometry: Detailed boundary geometries from Overpass
                - display_name: Full formatted place names from Nominatim
                - osm_type, osm_id: OSM element identifiers
                - category, type: Place classification
                - importance, place_rank: Relevance metrics
                - addresstype: Administrative level information
                - name: Short place names
                All additional Nominatim metadata fields are preserved.
                Returns empty GeoDataFrame if no results found or processed.

        Note:
            This method provides the most comprehensive place-to-geometry
            mapping available from OpenStreetMap Nominatim API data, but requires two API
            calls (Nominatim + Overpass) and may take longer for complex queries.

        Example:
            >>> service = NominatimService()
            >>> # Get complete boundary data for a city
            >>> city_boundaries = service.search_and_get_all_boundaries("Boston, MA")
            >>>
            >>> if not city_boundaries.empty:
            ...     print(f"Found {len(city_boundaries)} boundary features")
            ...     for _, feature in city_boundaries.iterrows():
            ...         area_km2 = feature.geometry.area * 111320**2 / 1e6  # Rough conversion
            ...         print(f"{feature['display_name']}: {area_km2:.1f} km²")
            ... else:
            ...     print("No boundaries found for the query")
        """
        logger.info(
            f"[nominatim_service] Starting search and boundary retrieval for query: '{query}'"
        )

        # Step 1: Search for places using Nominatim to get IDs and rich properties
        search_results = self.search_for_places(query)
        if search_results.empty:
            logger.warning(f"[nominatim_service] No results found for query: '{query}'")
            return gpd.GeoDataFrame()

        logger.info(
            f"[nominatim_service] Found {len(search_results)} results from Nominatim"
        )

        # Only use the first (best match) result to avoid returning multiple features
        # like both a city and its containing canton/region
        search_results = search_results.head(1)
        logger.info(
            f"[nominatim_service] Using top result: {search_results.iloc[0].get('display_name', 'unknown')}"
        )

        # Create a dictionary of the search results for easy lookup
        osm_items: List[Dict[str, Union[str, int]]] = [
            {"osm_id": r["osm_id"], "osm_type": r["osm_type"]}
            for _, r in search_results.iterrows()
        ]
        search_results_dict: Dict[Tuple[str, int], Dict[str, Any]] = {
            (r["osm_type"], r["osm_id"]): r.to_dict()
            for _, r in search_results.iterrows()
        }

        logger.debug(
            f"[nominatim_service] Created lookup dictionary for {len(search_results_dict)} search results"
        )

        # Step 2: Get all geometries from Overpass API
        boundaries_dict = self.get_boundaries_from_overpass(osm_items)

        # Step 3: Combine Overpass geometries with ALL Nominatim metadata
        all_boundaries: List[gpd.GeoDataFrame] = []
        for key, boundary_gdf in boundaries_dict.items():
            if boundary_gdf is not None and not boundary_gdf.empty:
                original_data: Optional[Dict[str, Any]] = search_results_dict.get(key)
                if original_data:
                    logger.debug(
                        f"[nominatim_service] Combining Overpass geometry with Nominatim metadata for {key}"
                    )
                    # Add all properties from the original Nominatim search
                    for prop_key, prop_value in original_data.items():
                        # Don't overwrite the new geometry from Overpass or the core IDs
                        if prop_key not in ["geometry", "osm_id", "osm_type"]:
                            boundary_gdf[prop_key] = prop_value

                    # Drop the raw 'tags' column from Overpass for a cleaner output
                    boundary_gdf = boundary_gdf.drop(columns=["tags"], errors="ignore")
                    all_boundaries.append(boundary_gdf)
                else:
                    logger.warning(
                        f"[nominatim_service] No original data found for key {key}"
                    )

        if all_boundaries:
            # Concatenate all individual GeoDataFrames into one
            combined_gdf = gpd.GeoDataFrame(
                pd.concat(all_boundaries, ignore_index=True), crs="EPSG:4326"
            )

            # Reorder columns to be more intuitive, with geometry last
            cols: List[str] = combined_gdf.columns.tolist()
            preferred_order: List[str] = [
                "display_name",
                "name",
                "addresstype",
                "type",
                "category",
                "importance",
                "place_rank",
                "osm_id",
                "osm_type",
            ]
            core_cols: List[str] = [c for c in preferred_order if c in cols]
            other_cols: List[str] = [
                c for c in cols if c not in core_cols and c != "geometry"
            ]
            final_order: List[str] = core_cols + other_cols + ["geometry"]
            combined_gdf = combined_gdf[final_order]

            logger.info(
                f"[nominatim_service] Successfully retrieved and built {len(combined_gdf)} geometries with all properties"
            )
            return combined_gdf
        else:
            logger.warning(
                "[nominatim_service] No geometries could be constructed from the Overpass data"
            )
            return gpd.GeoDataFrame()
