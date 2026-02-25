import logging
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


from services.sql_tool_utils import (
    add_limit_to_sql,
    handle_sql_tool_with_aoi,
)
from geoforge_services.utils.llm import secure_gemini_llm_call_with_structured_output

load_dotenv()

logger = logging.getLogger(__name__)


class ClassSelector(BaseModel):
    """Selected classes and subtypes that are relevant to the user query."""

    subtypes: List[str] = Field(
        description="List of subtype names that are relevant to the user query. Can be empty if no subtypes are relevant."
    )


def get_sql(features: str) -> Dict[str, List[str]]:

    subtype_list = ["road", "rail", "water"]

    subclass_list = [
        "link",
        "sidewalk",
        "crosswalk",
        "parking_aisle",
        "driveway",
        "alley",
        "cycle_crossing",
    ]

    subtype_list = "\n".join(subtype_list)
    subclass_list = "\n".join(subclass_list)
    logger.debug(
        "[overture_transportation_tool] Available subtypes and subclasses prepared"
    )

    system_prompt = f"""You are a specialized assistant that analyzes user queries and identifies which classes, subtypes, and subclasses are most relevant.
    AVAILABLE SUBTYPES:
    {subtype_list}


INSTRUCTIONS
You are an assistant that helps categorize transportation-related queries. Follow these instructions:

Carefully analyze the user's query to understand what they're asking about.
From the available lists, select the CLASSES, SUBTYPES, and SUBCLASSES that are most relevant to the query.
Be selective - only choose options that directly relate to the core of the user's query.
You can select zero, one, or multiple items from each category.
If no items in a category are relevant, leave that category empty.
Don't force selections when uncertain - it's better to leave a category empty than to include irrelevant options.
Match the specificity level of your selections to the query - general queries get general selections, specific queries get specific selections.        """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"all {features}"),
    ]

    logger.debug(
        "[overture_transportation_tool] Invoking LLM to select transportation subtypes"
    )
    response = secure_gemini_llm_call_with_structured_output(messages, ClassSelector)
    logger.debug("[overture_transportation_tool] LLM response received")

    subtypes = response.subtypes if response.subtypes else []
    logger.debug(f"[overture_transportation_tool] Selected subtypes: {subtypes}")

    if not subtypes:
        logger.debug(
            "[overture_transportation_tool] No filters applied, using base SQL"
        )
        return "select * from overture_transportation"

    sql = "select * from overture_transportation\nwhere "

    conditions = []
    if subtypes:
        subtype_list_str = ", ".join([f"'{s}'" for s in subtypes])
        conditions.append(f"subtype IN ({subtype_list_str})")

    sql += " AND ".join(conditions)
    logger.debug("[overture_transportation_tool] Generated filtered SQL")

    return sql


def get_overture_transportation_tool(
    aoi_layer_id: str, feature_name: Optional[str] = None
) -> str:
    """
    Retrieves data for 'Overture transportation'.
    Requires an AOI layer ID and an optional feature name.
    """
    logger.info(
        f"[overture_transportation_tool] Retrieving Overture transportation data for AOI: {aoi_layer_id}"
    )
    try:
        if not feature_name:
            sql = "select * from overture_transportation"
            logger.debug(
                "[overture_transportation_tool] No feature name provided, using default SQL query"
            )
        else:
            sql = get_sql(feature_name)
            logger.debug(
                f"[overture_transportation_tool] Generated SQL query for feature: {feature_name}"
            )

        max_limit = 500_000_000

        sql = add_limit_to_sql(sql, max_limit)

        layers_data = handle_sql_tool_with_aoi(sql, aoi_layer_id)
        logger.info(
            f"[overture_transportation_tool] Successfully retrieved Overture transportation data: {len(layers_data)} layer(s)"
        )
        return layers_data
    except Exception as e:
        logger.error(
            f"[overture_transportation_tool] Failed to retrieve Overture transportation data: {e}",
            exc_info=True,
        )
        raise
