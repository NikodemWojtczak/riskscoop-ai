import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


from services.sql_tool_utils import (
    add_limit_to_sql,
    handle_sql_tool_with_aoi,
)

load_dotenv()
logger = logging.getLogger(__name__)


class ClassSelector(BaseModel):
    """Selected classes and subtypes that are relevant to the user query."""

    classes: List[str] = Field(
        description="List of class names that are relevant to the user query. Can be empty if no classes are relevant."
    )
    subtypes: List[str] = Field(
        description="List of subtype names that are relevant to the user query. Can be empty if no subtypes are relevant."
    )


def get_sql(features: str) -> Dict[str, List[str]]:

    # Extract page_c
    subtype_list = [
        "aerialway",
        "airport",
        "barrier",
        "bridge",
        "communication",
        "manhole",
        "pedestrian",
        "pier",
        "power",
        "recreation",
        "tower",
        "transit",
        "utility",
        "waste_management",
        "water",
    ]

    subtype_list = "\n".join(subtype_list)
    system_prompt = f"""You are a specialized assistant that analyzes user queries and identifies which classes and subtypes are most relevant.

    AVAILABLE CLASSES:
    fence
crossing
parking_space
hedge
gate
parking
power_pole
bench
waste_basket
street_lamp
stop
give_way
fire_hydrant
bus_stop
wall
traffic_signals
bridge
platform
recycling
bicycle_parking
kerb
substation
information
stop_position
bollard
parking_entrance
power_tower
lift_gate
pier
minor_line
retaining_wall
block
fountain
vending_machine
motorcycle_parking
drinking_water
generator
power_line
switch
cycle_barrier
insulator
cable
post_box
toilets
waste_disposal
breakwater
storage_tank
barrier
communication_tower
taxiway
pylon
street_cabinet
swing_gate
entrance
height_restrictor
charging_station
viewpoint
atm
pipeline
utility_pole
portal
motorway_junction
lighting
reservoir_covered
guard_rail
transformer
aerialway_station
bicycle_rental
chain
airport_gate
catenary_mast
ferry_terminal
handrail
border_control
boardwalk
helipad
bell_tower
terminal
railway_station
jersey_barrier
mobile_phone_tower
diving
hampshire_gate
viaduct
silo
weir
cattle_grid
kissing_gate
railway_halt
ditch
city_wall
plant
full-height_turnstile
apron
water_tower
stile
observation
platter
manhole
planter
runway
toll_booth
milestone
connection
defensive
radar
drag_lift
bus_station
dam
sally_port
bump_gate
chair_lift
bridge_support
mixed_lift
bus_trap
gondola
stopway
cable_car
j-bar
private_airport
camp_site
monitoring
rope_tow
airstrip
covered
regional_airport
zip_line
international_airport

    AVAILABLE SUBTYPES:
    {subtype_list}


TASK:
1. Analyze the user's query carefully and select the most relevant.
3. Determine which road-related class or classes, which road-related subtype or subtypes, and which road-related subclass or subclasses from the available lists are most relevant to answering the query.
4. You can select zero, one, or multiple classes, subtypes, and subclasses.
5. If no road-related classes, subtypes, or subclasses are relevant, return an empty list for that field.
6. Don't fill the classes, subtypes, or subclasses field if there are none relevant to roads in the query.
7. Avoid unnecessary classes and subtypes, only select the most relevant ones related to roads.
8. Try to be as general as possible. if user selects roads just select roads and nothing else.

Return your response in the specified structured format with the list of road-related classes, subtypes, and subclasses.
    """

    # Initialize Gemini LLM with structured output
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    # Use with_structured_output for Pydantic model
    structured_llm = llm.with_structured_output(ClassSelector)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"all {features}"),
    ]

    logger.debug(
        "[overture_infrastructure_tool] Invoking LangChain Gemini to select classes and subtypes"
    )
    response = structured_llm.invoke(messages)
    logger.debug("[overture_infrastructure_tool] LLM response received")

    classes = response.classes if response.classes else []
    subtypes = response.subtypes if response.subtypes else []

    if not classes and not subtypes:
        # If no relevant classes or subtypes, return the SQL query without filters
        return "select * from overture_infrastructure"

    sql = "select * from overture_infrastructure\nwhere "

    conditions = []
    if classes:
        class_list_str = ", ".join([f"'{c}'" for c in classes])
        conditions.append(f"class IN ({class_list_str})")
    if subtypes:
        subtype_list_str = ", ".join([f"'{s}'" for s in subtypes])
        conditions.append(f"subtype IN ({subtype_list_str})")

    sql += " AND ".join(conditions)

    return sql


def get_overture_infrastructure_tool(
    aoi_layer_id: str, feature_name: Optional[str] = None
) -> str:
    """
    Retrieves data for 'Overture infrastructure'.
    Requires an AOI layer ID and an optional feature name.
    """
    logger.info(
        f"Executing get_overture_infrastructure_tool with aoi_layer_id: {aoi_layer_id}, feature_name: {feature_name}"
    )
    try:
        # Placeholder: Actual data retrieval logic would go here
        if not feature_name:
            sql = "select * from overture_infrastructure"
            logger.debug(
                "No feature_name provided, using default SQL query for Overture infrastructure."
            )
        else:
            sql = get_sql(feature_name)
            logger.debug(
                f"Generated SQL query for Overture infrastructure with feature_name '{feature_name}': {sql}"
            )

        max_limit = 500_000_000

        sql = add_limit_to_sql(sql, max_limit)

        layers_data = handle_sql_tool_with_aoi(sql, aoi_layer_id)
        logger.debug(
            f"Successfully retrieved Overture infrastructure data for aoi_layer_id: {aoi_layer_id}, feature_name: {feature_name}"
        )
        return layers_data
    except Exception as e:
        logger.error(
            f"Error in get_overture_infrastructure_tool for aoi_layer_id: {aoi_layer_id}, feature_name: {feature_name}: {e}",
            exc_info=True,
        )
        raise
