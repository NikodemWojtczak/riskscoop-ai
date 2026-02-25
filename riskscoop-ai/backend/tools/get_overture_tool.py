"""Overture Maps Data Retrieval Tool.

This tool executes SQL queries against Overture Maps Foundation datasets
within a specified Area of Interest (AOI). The agent is responsible for
constructing the appropriate SQL query based on user requests.

Available Tables:
- overture_buildings: Building footprints with class/subtype columns
- overture_places: Points of interest with primary_category column
- overture_infrastructure: Infrastructure with class/subtype columns
- overture_transportation: Transportation network with subtype column

Usage Flow:
1. First create an AOI using get_division tool
2. Agent constructs SQL query based on user request and available columns
3. Use get_overture_data to execute query within AOI
4. Multiple queries can run in parallel (one per table)
"""

from agno.tools import tool
from agno.run import RunContext

from services.layer_service import LayerService
from services.sql_tool_utils import execute_sql_with_aoi

layer_service = LayerService()


@tool
def get_overture_data(
    run_context: RunContext,
    aoi_layer_name: str,
    sql: str,
    output_layer_name: str,
) -> str:
    """Execute SQL query on Overture Maps data within an Area of Interest.

    This tool executes a SQL query against Overture Maps tables and returns
    only features that intersect with the specified AOI polygon. Multiple
    invocations can run in parallel for different tables.

    Available Tables:
    - overture_buildings: columns include geometry, class, subtype, height, etc.
    - overture_places: columns include geometry, primary_category, name, etc.
    - overture_infrastructure: columns include geometry, class, subtype, etc.
    - overture_transportation: columns include geometry, subtype, class, etc.

    Args:
        session_state: The session state dictionary containing existing layers.
        aoi_layer_name: Name of the AOI layer in session_state["layers"].
            Must be created first using get_division tool.
        sql: SQL SELECT query to execute. Should query one of the Overture tables.
            Examples:
            - "SELECT * FROM overture_buildings WHERE subtype = 'medical'"
            - "SELECT * FROM overture_places WHERE primary_category = 'restaurant'"
            - "SELECT * FROM overture_buildings WHERE class IN ('hospital', 'school')"
            - "SELECT * FROM overture_transportation WHERE subtype = 'road'"
        output_layer_name: Name for the output layer in session_state.
            Use descriptive names like "hospitals", "restaurants", "roads".

    Returns:
        str: Success message with feature count, or error message.

    Example:
        >>> # Query hospitals
        >>> get_overture_data(session_state, "warsaw_aoi",
        ...     "SELECT * FROM overture_buildings WHERE class = 'hospital'",
        ...     "hospitals")
        >>> # Query restaurants (can run in parallel with above)
        >>> get_overture_data(session_state, "warsaw_aoi",
        ...     "SELECT * FROM overture_places WHERE primary_category = 'restaurant'",
        ...     "restaurants")
    """
    session_state = run_context.session_state
    if session_state is None:
        session_state = {}

    # Get AOI layer UUID from session state
    if "layers" not in session_state or aoi_layer_name not in session_state["layers"]:
        available = list(session_state.get("layers", {}).keys())
        if available:
            return f"ERROR: AOI layer '{aoi_layer_name}' not found. Available layers: {available}. Use one of these or create a new AOI with get_division first."
        else:
            return f"ERROR: No layers exist yet! You MUST call get_division FIRST to create an AOI layer before calling get_overture_data. Example: get_division(session_state, 'Warsaw, Poland', 'aoi')"

    aoi_layer_uuid = session_state["layers"][aoi_layer_name]

    # Execute query with AOI intersection
    gdf = execute_sql_with_aoi(
        sql=sql,
        aoi_layer_uuid=aoi_layer_uuid,
        session_state=session_state,
        output_layer_name=output_layer_name,
    )

    if gdf.empty:
        return f"No features found for query. Saved empty layer '{output_layer_name}'."

    return f"Retrieved {len(gdf)} features. Saved as layer '{output_layer_name}'."


@tool
def list_layers(run_context: RunContext) -> str:
    """List all available layers in the current session.

    Args:
        session_state: The session state dictionary containing layers.

    Returns:
        str: Formatted list of layer names and their UUIDs.
    """
    session_state = run_context.session_state
    if session_state is None:
        session_state = {}

    if "layers" not in session_state or not session_state["layers"]:
        return "No layers in session. Use get_division to create an AOI first."

    layers = session_state["layers"]
    lines = ["Available layers:"]
    for name, uuid in layers.items():
        lines.append(f"  - {name}: {uuid[:8]}...")

    return "\n".join(lines)
