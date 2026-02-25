"""Overture Buildings Dataset Service.

This module provides access to the Overture Maps Foundation buildings dataset
with semantic filtering capabilities using LLM for intelligent class/subtype selection.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import geopandas as gpd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from services.sql_tool_utils import add_limit_to_sql, execute_sql_with_aoi
from services.layer_service import LayerService

load_dotenv()
logger = logging.getLogger(__name__)

_layer_service = LayerService()

# Available building subtypes from Overture schema
BUILDING_SUBTYPES = [
    "agricultural", "civic", "commercial", "education", "entertainment",
    "industrial", "medical", "military", "outbuilding", "religious",
    "residential", "service", "transportation",
]

# Available building classes from Overture schema
BUILDING_CLASSES = [
    "apartments", "house", "roof", "shed", "detached", "garage", "service",
    "residential", "school", "greenhouse", "garages", "retail", "carport",
    "industrial", "office", "university", "semidetached_house", "commercial",
    "terrace", "farm_auxiliary", "church", "public", "warehouse", "hangar",
    "hospital", "allotment_house", "hotel", "barn", "farm", "hut", "government",
    "parking", "chapel", "bungalow", "sports_centre", "semi", "train_station",
    "toilets", "civic", "transformer_tower", "kindergarten", "transportation",
    "college", "post_office", "cabin", "fire_station", "supermarket", "library",
    "sports_hall", "temple", "stable", "guardhouse", "dormitory", "bunker",
    "ger", "grandstand", "stadium", "static_caravan", "cowshed", "manufacture",
    "mosque", "outbuilding", "synagogue", "storage_tank", "kiosk", "slurry_tank",
    "pavilion", "cathedral",
]


class ClassSelector(BaseModel):
    """Model for selecting relevant building classes and subtypes from user queries."""

    classes: List[str] = Field(
        default_factory=list,
        description="List of building class names relevant to the user query."
    )
    subtypes: List[str] = Field(
        default_factory=list,
        description="List of building subtype names relevant to the user query."
    )


def _get_class_selector_prompt() -> str:
    """Generate the system prompt for class/subtype selection."""
    return f"""You are a specialized assistant that analyzes user queries and identifies which building classes and subtypes are most relevant.

AVAILABLE CLASSES:
{chr(10).join(BUILDING_CLASSES)}

AVAILABLE SUBTYPES:
{chr(10).join(BUILDING_SUBTYPES)}

TASK:
1. Analyze the user's query carefully.
2. Select relevant classes and/or subtypes from the available lists.
3. Only select items directly relevant to the query.
4. Leave fields empty if uncertain or if the query is too general (e.g., "all buildings").
5. Example: "schools and hospitals" -> subtypes: ["education", "medical"]
6. Example: "buildings in the area" -> leave both empty

Return your response in the structured format with lists of classes and subtypes."""


def get_building_sql(features: Optional[str] = None) -> str:
    """Generate SQL query for Overture buildings with semantic filtering.

    Uses LLM to map natural language feature descriptions to appropriate
    SQL filters on building classes and subtypes.

    Args:
        features: Natural language description of desired building types.
            Examples: 'schools', 'commercial buildings', 'hospitals'

    Returns:
        SQL query string with appropriate WHERE clause.
    """
    base_sql = "SELECT * FROM overture_buildings"

    if not features:
        logger.debug("[buildings] No features specified, returning base SQL")
        return base_sql

    logger.info(f"[buildings] Generating SQL for features: {features}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ClassSelector)

    messages = [
        SystemMessage(content=_get_class_selector_prompt()),
        HumanMessage(content=f"Find: {features}"),
    ]

    response = structured_llm.invoke(messages)
    classes = response.classes or []
    subtypes = response.subtypes or []

    logger.debug(f"[buildings] Selected classes: {classes}, subtypes: {subtypes}")

    if not classes and not subtypes:
        return base_sql

    conditions = []
    if classes:
        class_list = ", ".join([f"'{c}'" for c in classes])
        conditions.append(f"class IN ({class_list})")
    if subtypes:
        subtype_list = ", ".join([f"'{s}'" for s in subtypes])
        conditions.append(f"subtype IN ({subtype_list})")

    return f"{base_sql} WHERE {' OR '.join(conditions)}"


def get_overture_buildings(
    aoi_layer_uuid: str,
    session_state: Dict[str, Any],
    output_layer_name: str,
    feature_name: Optional[str] = None,
    max_limit: int = 500_000,
) -> str:
    """Retrieve building footprints from Overture Maps within an AOI.

    Args:
        aoi_layer_uuid: UUID of the AOI layer for spatial filtering.
        session_state: Session state dictionary to store the result layer.
        output_layer_name: Name for the output layer in session state.
        feature_name: Optional natural language description of building types.
        max_limit: Maximum number of features to return.

    Returns:
        UUID of the saved result layer.

    Example:
        >>> uuid = get_overture_buildings(
        ...     aoi_uuid,
        ...     session_state,
        ...     "hospitals_layer",
        ...     "hospitals and medical facilities"
        ... )
    """
    logger.info(f"[buildings] Retrieving buildings for AOI: {aoi_layer_uuid}")

    sql = get_building_sql(feature_name)
    sql = add_limit_to_sql(sql, max_limit)

    gdf = execute_sql_with_aoi(
        sql=sql,
        aoi_layer_uuid=aoi_layer_uuid,
        session_state=session_state,
        output_layer_name=output_layer_name,
    )

    if gdf.empty:
        logger.warning("[buildings] No buildings found in AOI")
        return ""

    # Return the UUID from session state
    return session_state["layers"][output_layer_name]
