"""Administrative Division Boundary Tool.

This tool retrieves administrative division boundaries (countries, states, cities,
districts, etc.) using OpenStreetMap Nominatim and Overpass APIs. It's the first
step in the geospatial analysis workflow - establishing an Area of Interest (AOI).

Usage Flow:
1. User requests a geographic area (e.g., "Warsaw, Poland")
2. Tool searches Nominatim for matching places
3. Retrieves detailed boundary geometry from Overpass API
4. Saves as GeoJSON layer and stores UUID in session_state
5. This layer can then be used as AOI for subsequent Overture data queries
"""

from agno.tools import tool
from agno.run import RunContext

from services.nominatim_service import NominatimService
from services.layer_service import LayerService

nominatim_service = NominatimService()
layer_service = LayerService()


@tool
def get_division(
    run_context: RunContext,
    feature_query: str,
    layer_name: str,
) -> str:
    """Retrieve administrative division boundaries and save as a layer.

    This tool searches for geographic administrative divisions (countries, states,
    cities, districts, neighborhoods, etc.) using OpenStreetMap data and saves
    the boundary geometry as a GeoJSON layer. The resulting layer can be used
    as an Area of Interest (AOI) for subsequent Overture Maps data queries.

    Supported Query Types:
    - Countries: "France", "Germany", "Poland"
    - States/Regions: "California, USA", "Bavaria, Germany", "Mazowieckie, Poland"
    - Cities: "Paris, France", "Warsaw, Poland", "New York City"
    - Districts: "Manhattan, New York", "Praga-Północ, Warsaw"
    - Neighborhoods: "Montmartre, Paris", "Śródmieście, Warsaw"
    - Natural features: "Central Park, NYC", "Thames River, London"

    Args:
        session_state: The session state dictionary. The layer UUID will be stored
            at session_state["layers"][layer_name] for use by other tools.
        feature_query: Natural language search query for the geographic feature.
            Should include enough context for unambiguous identification.
            Examples:
            - "Warsaw, Poland" (city)
            - "Mazowieckie, Poland" (voivodeship/region)
            - "Manhattan, New York, USA" (borough)
            - "Central Park, New York" (park)
        layer_name: Identifier name for this layer in the session state.
            Use descriptive names like "aoi", "study_area", "city_boundary".
            This name will be used to reference the layer in subsequent operations.

    Returns:
        str: The UUID of the saved GeoJSON layer file, or an error message
            if no boundaries were found for the query.

    Example:
        >>> # Create an AOI for Warsaw
        >>> uuid = get_division(session_state, "Warsaw, Poland", "warsaw_aoi")
        >>> # Now session_state["layers"]["warsaw_aoi"] contains the UUID
        >>> # This AOI can be used with get_overture_data tool

    Note:
        - The tool returns the first/best matching result from Nominatim
        - For ambiguous queries, add more context (country, region)
        - Boundaries are retrieved in WGS84 (EPSG:4326) coordinate system
        - Complex boundaries (multipolygons) are fully supported
    """
    session_state = run_context.session_state
    if session_state is None:
        session_state = {}

    gdf = nominatim_service.search_and_get_all_boundaries(feature_query)

    if gdf.empty:
        return f"No boundaries found for query: '{feature_query}'. Try adding more context (e.g., country name) or check spelling."

    file_uuid = layer_service.save_layer(gdf, session_state, layer_name)

    return f"Successfully created layer '{layer_name}' with {len(gdf)} feature(s). UUID: {file_uuid}"
