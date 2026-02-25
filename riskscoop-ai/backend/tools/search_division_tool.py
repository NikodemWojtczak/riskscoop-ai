"""Search Division Tool.

This tool searches for geographic places using OpenStreetMap Nominatim API
and returns a list of matching results. The agent can then present these
options to the user and use get_division to retrieve the selected boundary.

Usage Flow:
1. User provides an ambiguous location query
2. Tool searches Nominatim and returns matching places with metadata
3. Agent presents options to user (display_name, type, importance)
4. User selects the desired option
5. Agent calls get_division with the specific query
"""

from agno.tools import tool

from services.nominatim_service import NominatimService

nominatim_service = NominatimService()


@tool
def search_division(
    feature_query: str,
) -> str:
    """Search for geographic places and return matching results.

    This tool searches OpenStreetMap Nominatim for places matching the query
    and returns a list of results with metadata. Use this when:
    - The user's query might be ambiguous (e.g., "Springfield" - many cities)
    - You want to show the user available options before selecting
    - You need to verify which specific place the user means

    The results include:
    - display_name: Full formatted name with administrative hierarchy
    - type: Place type (city, town, village, administrative, etc.)
    - category: OSM category (boundary, place, etc.)
    - importance: Relevance score (0-1, higher = more important/well-known)
    - osm_type: OSM element type (node, way, relation)
    - osm_id: Unique OSM identifier

    Args:
        feature_query: Natural language search query for the geographic feature.
            Examples:
            - "Warsaw" (might return multiple results)
            - "Springfield" (many cities with this name)
            - "Paris" (city in France, but also in Texas, etc.)

    Returns:
        str: Formatted list of matching places with metadata, or an error
            message if no results found.

    Example:
        >>> # Search for places named "Springfield"
        >>> results = search_division("Springfield")
        >>> # Returns list of all Springfield cities with their states/countries
        >>> # Agent can then present options to user

    Note:
        - Results are sorted by importance (most relevant first)
        - Maximum 50 results returned
        - After user selects, use get_division to retrieve the boundary
    """
    gdf = nominatim_service.search_for_places(feature_query)

    if gdf.empty:
        return f"No places found for query: '{feature_query}'. Try a different spelling or add more context."

    # Format results for the agent to present to user
    results = []
    for idx, row in gdf.iterrows():
        result = {
            "index": idx + 1,
            "display_name": row.get("display_name", "Unknown"),
            "type": row.get("type", "Unknown"),
            "category": row.get("category", "Unknown"),
            "importance": round(row.get("importance", 0), 4),
            "osm_type": row.get("osm_type", "Unknown"),
            "osm_id": row.get("osm_id", "Unknown"),
        }
        results.append(result)

    # Build formatted output
    output_lines = [f"Found {len(results)} places matching '{feature_query}':\n"]

    for r in results[:15]:  # Limit to top 15 for readability
        output_lines.append(
            f"{r['index']}. {r['display_name']}\n"
            f"   Type: {r['type']} | Category: {r['category']} | Importance: {r['importance']}"
        )

    if len(results) > 15:
        output_lines.append(f"\n... and {len(results) - 15} more results.")

    output_lines.append(
        "\n\nTo select a place, use get_division with the full display_name or a more specific query."
    )

    return "\n".join(output_lines)
