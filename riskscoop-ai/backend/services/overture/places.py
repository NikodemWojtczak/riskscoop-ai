"""Overture Places Dataset Tool.

This module provides access to the Overture Maps Foundation places dataset through
the Geoforge platform. The Overture Places dataset contains comprehensive point-based
location data including businesses, landmarks, civic buildings, and points of interest
with detailed classification and attribute information.

Key Features:
- Point geometries representing places of interest
- Primary category classification (food, retail, healthcare, education, etc.)
- Detailed place attributes including names, addresses, and business information
- RAG-powered intelligent category selection
- Spatial filtering by Area of Interest (AOI)
- SQL query generation with semantic understanding

Data Source:
The Overture Maps Foundation provides open, interoperable map data including places
derived from multiple authoritative sources. Place data covers businesses, civic
facilities, landmarks, services, and other points of interest with rich metadata.

Integration:
- MotherDuck database backend for high-performance spatial queries
- RAG service for semantic understanding of place categories
- Gemini LLM for intelligent primary category selection
- Spatial tools for AOI-based filtering

Example Categories:
- Food & Dining: restaurants, cafes, bars, fast food
- Retail & Services: shops, banks, post offices, gas stations
- Healthcare: hospitals, clinics, pharmacies
- Education: schools, universities, libraries
- Transportation: airports, train stations, bus stops
"""

import logging
import os
from typing import List, Optional

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


class PrimaryCategorySelector(BaseModel):
    """Model for selecting relevant place primary categories from user queries.

    This model is used to parse LLM responses when determining which place
    primary categories are relevant to a user's query. It enables semantic
    understanding of requests like 'restaurants and cafes' to be mapped to
    appropriate primary categories like 'food_and_drink'.

    Attributes:
        primary_category: Place primary category names from the Overture schema
                         (e.g., 'food_and_drink', 'retail', 'healthcare', 'education')
    """

    primary_category: List[str] = Field(
        description="List of primary category names relevant to the user query. "
        "Can be empty if no specific categories are relevant or for general POI queries."
    )


def get_sql(features: str) -> str:
    """Generate a SQL query for Overture places based on semantic feature understanding.

    Uses RAG (Retrieval Augmented Generation) to understand available place primary
    categories and then leverages LLM to semantically map user requests to appropriate
    SQL filters. This enables natural language queries like 'restaurants and cafes'
    to be converted to proper WHERE clauses filtering on primary categories.

    Args:
        features: Natural language description of desired place types
                 (e.g., 'restaurants', 'hospitals', 'schools', 'banks')

    Returns:
        SQL query string with appropriate WHERE clause for place filtering.
        Returns base query 'select * from overture_places' if no specific
        categories can be determined or for general queries.

    Example:
        >>> get_sql('restaurants and cafes')
        "select * from overture_places\\nwhere primary_category IN ('food_and_drink')"
    """
    logger.info(f"[overture_places_tool] Generating SQL for features: {features}")
    logger.debug(
        "[overture_places_tool] Building system prompt with available primary categories"
    )

    # System prompt with all available primary categories (hardcoded list)
    system_prompt = """
You are a specialized assistant that analyzes user queries and identifies which primary_category and subtypes are most relevant.

AVAILABLE primary_category:
community_services_non_profits
professional_services
beauty_salon
beauty_and_spa
restaurant
hotel
automotive_repair
hair_salon
jewelry_store
french_restaurant
clothing_store
gym
spas
real_estate
pharmacy
bar
real_estate_agent
event_planning
school
shopping
cafe
appellate_practice_lawyers
active_life
naturopathic_holistic
bakery
elementary_school
pizza_restaurant
business_management_services
grocery_store
physical_therapy
italian_restaurant
hospital
landmark_and_historical_building
contractor
art_gallery
health_and_medical
church_cathedral
womens_clothing_store
travel_services
construction_services
supermarket
bank_credit_union
fast_food_restaurant
sports_club_and_league
doctor
education
car_dealer
financial_service
flowers_and_gifts_shop
park
college_university
dentist
furniture_store
advertising_agency
arts_and_entertainment
engineering_services
coffee_shop
non_governmental_association
home_cleaning
psychologist
gas_station
counseling_and_mental_health
internal_medicine
insurance_agency
transportation
yoga_studio
winery
eyewear_and_optician
shoe_store
accountant
public_and_government_association
theatre
architectural_designer
dance_club
town_hall
tattoo_and_piercing
retail
pub
sports_and_recreation_venue
psychiatrist
dance_school
information_technology_company
public_service_and_government
plumbing
diner
embassy
boutique
religious_organization
marketing_agency
topic_concert_venue
bicycle_shop
escrow_services
martial_arts_club
printing_services
veterinarian
liquor_store
sushi_restaurant
interior_design
painting
sporting_goods
marketing_consultant
architect
library
psychotherapist
driving_school
life_coach
nutritionist
travel
electrician
indian_restaurant
automotive
parking
butcher_shop
cosmetic_and_beauty_supplies
electronics
thai_restaurant
caterer
social_service_organizations
childrens_clothing_store
carpenter
car_rental_agency
music_school
chinese_restaurant
stadium_arena
barber
lake
podiatrist
language_school
tea_room
alternative_medicine
it_service_and_computer_repair
lawyer
home_improvement_store
cocktail_bar
ev_charging_station
burger_restaurant
massage_therapy
graphic_designer
pediatrician
arts_and_crafts
acupuncture
taxi_service
general_dentistry
bookstore
mens_clothing_store
web_designer
train_station
roofing
building_contractor
tobacco_shop
hvac_services
motorcycle_dealer
beach
shopping_center
landscaping
beauty_product_supplier
financial_advising
employment_agencies
pilates_studio
chocolatier
fashion
fashion_accessories_store
atms
key_and_locksmith
osteopathic_physician
farm
lounge
building_supply_store
museum
home_goods_store
industrial_equipment
obstetrician_and_gynecologist
music_venue
notary_public
telecommunications_company
laboratory_testing
software_development
property_management
central_government_office
attractions_and_activities
home_service
bridge
osteopath
landscape_architect
reflexology
cultural_center
post_office
hvac_supplier
sewing_and_alterations
airport
mountain
pet_store
paint_store
fitness_trainer
health_insurance_office
hardware_store
public_plaza
swimming_pool
pet_groomer
cinema
toy_store
laundromat
lebanese_restaurant
event_photography
computer_hardware_company
accommodation
nursery_and_gardening
retirement_home
bus_station
wine_bar
health_food_store
organization
massage
industrial_company
gardener
dry_cleaning
pathologist
flooring_contractors
process_servers
pool_cleaning
storage_facility
music_production
nail_salon
metal_fabricator
convenience_store
community_center
plastic_surgeon
home_health_care
textile_mill
resort
truck_dealer_for_businesses
delicatessen
chiropractor
amateur_sports_team
automotive_parts_and_accessories
charity_organization
korean_grocery_store
used_car_dealer
asian_restaurant
cardiologist
farmers_market
jewelry_and_watches_manufacturer
courier_and_delivery_services
travel_agents
ice_cream_shop
wholesale_store
monument
ophthalmologist
carpet_store
masonry_contractors
commercial_industrial
government_services
janitorial_services
private_school
tapas_bar
brewery
high_school
tire_dealer_and_repair
mobile_phone_store
medical_service_organizations
eat_and_drink
shipping_center
thrift_store
organic_grocery_store
electrical_supply_store
auto_detailing
musical_instrument_store
department_store
discount_store
catholic_church
fruits_and_vegetables
optometrist
japanese_restaurant
kitchen_supply_store
beer_garden
internet_marketing_service
political_organization
surgeon
motorcycle_repair
currency_exchange
medical_center
appliance_store
freight_and_cargo_service
broadcasting_media_production
diagnostic_imaging
portuguese_restaurant
lighting_store
trusts
drywall_services
real_estate_service
lingerie_store
art_school
business
fire_department
orthopedist
limo_services
investing
agriculture
pet_services
telecommunications
car_wash
courthouse
auto_electrical_repair
preschool
translating_and_interpreting_services
antique_store
auto_company
photography_store_and_services
tennis_court
turkish_restaurant
international_business_and_trade_services
florist
family_practice
food_truck
steakhouse
sandwich_shop
holiday_rental_home
amusement_park
banks
swiss_restaurant
cheese_shop
internet_service_provider
funeral_services_and_cemeteries
orthodontist
horse_boarding
ski_resort
dermatologist
altering_and_remodeling_contractor
local_and_state_government_offices
business_manufacturing_and_supply
home_security
tutoring_center
water_treatment_equipment_and_services
youth_organizations
river
legal_services
computer_store
business_advertising
day_care_preschool
topic_publisher
e_cigarette_store
private_investigation
agricultural_service
security_services
food_delivery_service
dog_trainer
peruvian_restaurant
structure_and_geography
health_spa
gift_shop
police_department
european_restaurant
middle_eastern_restaurant
civic_center
middle_school
art_museum
child_care_and_day_care
mexican_restaurant
movers
vietnamese_restaurant
boat_rental_and_training
escape_rooms
appraisal_services
business_consulting
evangelical_church
campground
commercial_real_estate
psychic
home_decor
campus_building
media_news_company
social_and_human_services
eye_care_clinic
radiologist
fondue_restaurant
radio_station
seafood_restaurant
newspaper_and_magazines_store
food
garbage_collection_service
pharmaceutical_products_wholesaler
labor_union
online_shop
pet_breeder
environmental_conservation_organization
home_developer
tours
civil_engineers
office_equipment
food_beverage_service_distribution
business_to_business
airport_terminal
security_systems
sunglasses_store
auditorium
mortgage_broker
windows_installation
automotive_dealer
senior_citizen_services
mattress_store
auto_body_shop
energy_company
vocational_and_technical_school
energy_equipment_and_solution
mediterranean_restaurant
public_relations
hearing_aids
tile_store
shoe_repair
doner_kebab
solar_installation
medical_research_and_development
equestrian_facility
oncologist
pancake_house
halfway_house
specialty_school
wedding_planning
spanish_restaurant
african_restaurant
private_association
glass_and_mirror_sales_service
ethiopian_restaurant
print_media
used_vintage_and_consignment
audio_visual_equipment_store
urologist
agricultural_cooperatives
boat_service_and_repair
ambulance_and_ems_services
mass_media
metals
rental_services
business_to_business_services
furniture_manufacturers
music_and_dvd_store
fountain
bridal_shop
candy_store
b2b_science_and_technology
linen
auto_glass_service
health_consultant
marina
skin_care
barbecue_restaurant
chicken_restaurant
neurologist
comic_books_store
bed_and_breakfast
money_transfer_services
occupational_therapy
bike_repair_maintenance
skate_park
party_and_event_planning
computer_coaching
patisserie_cake_shop
audiologist
self_storage_facility
hypnosis_hypnotherapy
boxing_class
casino
party_supply
kids_recreation_and_party
towing_service
biotechnology_company
moroccan_restaurant
it_consultant
hair_stylist
laser_hair_removal
engine_repair_service
latin_american_restaurant
movie_television_studio
pharmaceutical_companies
rental_service
tiling
beer_bar
hair_removal
fabric_store
wholesaler
ethical_grocery
auto_parts_and_supply_store
american_restaurant
pulmonologist
specialty_foods
castle
ferry_boat_company
breakfast_and_brunch_restaurant
mosque
auction_house
horseback_riding_service
ear_nose_and_throat
corporate_office
costume_store
audiovisual_equipment_rental
outlet_store
bubble_tea
masonry_concrete
e_commerce_service
hiking_trail
cabinet_sales_service
tax_services
vegetarian_restaurant
land_surveying
appliance_repair_service
beverage_store
dental_laboratories
electricity_supplier
airport_lounge
insulation_installation
venue_and_event_space
desserts
rheumatologist
automotive_services_and_repair
pediatric_dentist
outdoor_gear
nature_reserve
educational_research_institute
adult_education
psychic_medium
smoothie_juice_bar
pet_boarding
leather_goods
wine_wholesaler
ice_skating_rink
sign_making
damage_restoration
watch_store
sports_wear
pet_sitting
framing_store
auto_insurance
tanning_salon
shooting_range
home_and_garden
hobby_shop
farm_equipment_and_supply
pest_control_service
scuba_diving_center
cannabis_dispensary
public_school
animal_shelter
history_museum
food_consultant
hair_supply_stores
recycling_center
art_restoration_service
salad_bar
circus
weight_loss_center
asian_fusion_restaurant
auto_customization
educational_services
meditation_center
beer_wine_and_spirits
credit_and_debt_counseling
woodworking_supply_store
sports_and_fitness_instruction
vitamins_and_supplements
installment_loans
halal_restaurant
fire_protection_service
medical_spa
secretarial_services
donuts
fishmonger
fireplace_service
comedy_club
arcade
window_supplier
soccer_field
plastic_fabrication_company
video_game_store
media_agency
bowling_alley
buffet_restaurant
chimney_sweep
choir
computer_wholesaler
airline
medical_supply
motorsport_vehicle_dealer
texmex_restaurant
buddhist_temple
synagogue
adult_entertainment
hawaiian_restaurant
furniture_assembly
falafel_restaurant
business_office_supplies_and_stationery
disability_services_and_support_organization
image_consultant
oral_surgeon
performing_arts
luggage_store
personal_care_service
dim_sum_restaurant
forestry_service
flea_market
boat_dealer
public_health_clinic
business_schools
rock_climbing_spot
zoo
fair
airlines
race_track
environmental_conservation_and_ecological_organizations
marble_and_granite_professionals
nursing_school
gay_bar
farming_services
it_support_and_service
brazilian_restaurant
botanical_garden
golf_course
b2b_textiles
korean_restaurant
armed_forces_branch
hookah_bar
cosmetic_dentist
greek_restaurant
soccer_stadium
sports_medicine
furniture_accessory_store
gastroenterologist
water_park
medical_school
oil_and_gas
social_media_agency
excavation_service
human_resource_services
television_service_providers
argentine_restaurant
gun_and_ammo
grain_elevators
prenatal_perinatal_care
executive_search_consultants
hotel_bar
b2b_cleaning_and_waste_management
family_counselor
criminal_defense_law
divorce_and_family_law
anesthesiologist
countertop_installation
educational_camp
window_treatment_store
maternity_centers
waxing
horse_riding
tree_services
allergist
political_party_office
chemical_plant
motor_freight_trucking
superstore
bike_rentals
travel_company
session_photography
oil_refiners
lodge
bookkeeper
perfume_store
auto_restoration_services
mobile_phone_repair
brokers
cannabis_clinic
threading_service
crops_production
bar_and_grill_restaurant
hot_tubs_and_pools
endocrinologist
safe_store
gastropub
restaurant_wholesale
employment_law
patent_law
recreation_vehicle_repair
aircraft_dealer
dog_park
cosmetic_surgeon
handyman
drive_in_theater
passport_and_visa_services
home_staging
traditional_clothing
pool_billiards
bookbinding
beverage_supplier
food_stand
pier
karaoke
iron_and_steel_industry
videographer
mediator
hat_shop
poke_restaurant
metal_materials_and_experts
machine_shop
duty_free_shop
persian_iranian_restaurant
social_security_services
auto_manufacturers_and_distributors
shades_and_blinds
medical_transportation
hot_dog_restaurant
bagel_shop
golf_equipment
convents_and_monasteries
meat_wholesaler
b2b_jewelers
career_counseling
temp_agency
homeless_shelter
warehouses
vending_machine_supplier
science_museum
logging_equipment_and_supplies
builders
afghan_restaurant
surfing
valet_service
paintball
clothing_company
periodontist
chambers_of_commerce
trains
ip_and_internet_law
metal_supplier
playground
personal_chef
visitor_center
wholesale_grocer
customized_merchandise
theme_restaurant
cooking_school
book_magazine_distribution
distillery
rug_store
midwife
teeth_whitening
screen_printing_t_shirt_printing
public_utility_company
international_grocery_store
nurse_practitioner
hydraulic_equipment_supplier
3d_printing_service
lumber_store
cave
surgical_appliances_and_supplies
canoe_and_kayak_hire_service
national_park
aquarium
gold_buyer
colombian_restaurant
truck_rentals
taco_restaurant
car_window_tinting
art_restoration
dog_walkers
tennis_stadium
astrologer
night_market
books_mags_music_and_video
electrical_wholesaler
garage_door_service
vehicle_shipping
pasta_shop
sports_school
volunteer_association
ski_and_snowboard_shop
trophy_shop
indonesian_restaurant
observatory
business_records_storage_and_management
sri_lankan_restaurant
writing_service
jazz_and_blues
health_department
fur_clothing
guest_house
occupational_safety
immigration_law
lawn_service
test_preparation
cosmetic_products_manufacturer
sex_therapist
tire_shop
bedding_and_bath_stores
womens_health_clinic
first_aid_class
sightseeing_tour_agency
motel
cottage
nephrologist
emergency_room
brake_service_and_repair
diagnostic_services
forest
urgent_care_clinic
quay
driving_range
erotic_massage
miniature_golf_course
sky_diving
music_festivals_and_organizations
pawn_shop
cleaning_services
massage_school
community_gardens
coal_and_coke
surf_shop
talent_agency
cupcake_shop
kitchen_and_bath
scandinavian_restaurant
prosthetics
dj_service
emergency_service
truck_dealer
printing_equipment_and_supply
rv_rentals
atv_rentals_and_tours
water_purification_services
electronic_parts_supplier
urban_farm
dental_hygienist
crane_services
wood_and_pulp
water_heater_installation_repair
automotive_consultant
paving_contractor
exhibition_and_trade_center
theatrical_productions
wig_store
video_film_production
dairy_farm
stone_and_masonry
frozen_yoghurt_shop
sandblasting_service
cremation_services
wildlife_sanctuary
infectious_disease_specialist
recreational_vehicle_dealer
automotive_storage_facility
elevator_service
scuba_diving_instruction
truck_repair
pool_and_hot_tub_services
south_african_restaurant
record_label
foster_care_services
aromatherapy
armenian_restaurant
pentecostal_church
cycling_classes
commissioned_artist
golf_instructor
vinyl_record_store
hearing_aid_provider
commercial_printer
cleaning_products_supplier
pension
clinical_laboratories
grain_production
shredding_services
health_coach
flyboarding_center
pet_adoption
tax_law
professional_sports_team
scrap_metals
greengrocer
federal_government_offices
automobile_leasing
boat_tours
archery_shop
credit_union
bagel_restaurant
b2b_sporting_and_recreation_goods
bathroom_remodeling
pets
esports_league
collection_agencies
fertility
luggage_storage
hair_loss_center
piercing
onsen
dental_supply_store
poultry_farming
sculpture_statue
metal_plating_service
packing_supply
b2b_electronic_equipment
home_automation
chimney_service
patio_covers
homeopathic_medicine
hematology
mobile_home_dealer
b2b_apparel
auto_upholstery
airport_shuttles
health_food_restaurant
flight_school
ski_and_snowboard_school
lawn_mower_repair_service
plastic_manufacturer
commercial_vehicle_dealer
baptist_church
hospice
energy_management_and_conservation_consultants
international_restaurant
shutters
dive_bar
ironworkers
road_contractor
designer_clothing
glass_blowing
surgical_center
telephone_services
dairy_stores
polish_restaurant
home_inspector
skate_shop
optical_wholesaler
cement_supplier
bangladeshi_restaurant
health_market
trailer_dealer
photographer
hunting_and_fishing_supplies
junk_removal_and_hauling
cardiovascular_and_thoracic_surgeon
service_apartments
self_defense_classes
basketball_stadium
animation_studio
swimwear_store
carpet_cleaning
personal_injury_law
home_theater_systems_stores
cards_and_stationery_store
hindu_temple
rehabilitation_center
bartender
mining
home_organization
environmental_and_ecological_services_for_businesses
motorcycle_rentals
tattoo
hockey_arena
food_and_beverage_exporter
bail_bonds_service
boot_camp
logging_services
environmental_testing
wellness_program
contract_law
machine_and_tool_rentals
newspaper_advertising
rugby_stadium
dietitian
animal_rescue_service
boat_charter
debt_relief_services
housing_cooperative
uniform_store
wallpaper_store
go_kart_club
waterproofing
disability_law
anglican_church
pipeline_transportation
fishing_charter
hair_replacement
irrigation_companies
hostel
civilization_museum
noodles_restaurant
horse_equipment_shop
piano_store
automation_services
door_sales_service
japanese_confectionery_shop
fence_and_gate_sales_service
business_brokers
holding_companies
laboratory
lighthouse
automobile_registration_service
jail_and_prison
consultant_and_general_service
electric_utility_provider
professional_sports_league
importer_and_exporter
bus_tours
laser_eye_surgery_lasik
promotional_products_and_services
restaurant_equipment_and_supply
kitchen_remodeling
firewood
real_estate_investment
laser_tag
digitizing_services
russian_restaurant
irish_pub
lottery_ticket
aquatic_pet_store
computer_museum
b2b_furniture_and_housewares
electronics_repair_shop
nigerian_restaurant
pakistani_restaurant
bus_service
film_festivals_and_organizations
musician
data_recovery
health_and_wellness_club
inventory_control_service
himalayan_nepalese_restaurant
football_stadium
chinese_martial_arts_club
officiating_services
fire_and_water_damage_restoration
research_institute
department_of_motor_vehicles
cabin
skilled_nursing
packaging_contractors_and_service
blacksmiths
arabian_restaurant
public_market
clock_repair_service
b2b_medical_support_services
coach_bus
publicity_service
produce_wholesaler
gymnastics_center
iron_work
psychoanalyst
modern_art_museum
scooter_rental
rv_park
appliance_manufacturer
drone_store
boat_parts_and_supply_store
racquetball_court
rugby_pitch
horse_racing_track
mechanical_engineers
proctologist
opera_and_ballet
welding_supply_store
music_production_services
boat_storage_facility
welders
specialty_grocery_store
bulgarian_restaurant
drugstore
paralegal_services
aircraft_manufacturer
media_news_website
filipino_restaurant
mongolian_restaurant
animal_hospital
comfort_food_restaurant
swimming_instructor


TASK:
1. Analyze the user's query carefully.
2. Determine which primary_category or primary_categories from the available lists are most relevant to answering the query.
3. You can select zero, one, or multiple primary_category.
4. Only select primary_category that are directly relevant to the core of the user's query.
5. If no primary_category are relevant, return an empty list for that field.
7. For general queries like "poi" or "places" or "points of interest", leave the primary_category list empty.
6. get all the primary_category that are relevant to the query.
8. For specific queries like "schools", "hospitals", "grocery", select the appropriate matching categories from the available list.
9. Example: If user asks for "restaurants" or "firestation", you would select categories related to restaurants/food or firestation if they exist in the available list.
10. Example: If user asks for "places" or "poi" or "points of interest", you would leave the primary_category list empty as this is a general query.

Return your response in the specified structured format with the list of primary_category.
"""

    logger.debug("[overture_places_tool] System prompt prepared")

    # Initialize Gemini LLM with structured output
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    # Use with_structured_output for Pydantic model
    structured_llm = llm.with_structured_output(PrimaryCategorySelector)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"all {features}"),
    ]

    logger.debug(
        "[overture_places_tool] Invoking LangChain Gemini to select primary categories"
    )
    response = structured_llm.invoke(messages)
    logger.debug("[overture_places_tool] LLM response received")
    logger.debug(
        f"[overture_places_tool] Selected primary categories: {response.primary_category}"
    )

    primary_category = response.primary_category if response.primary_category else []

    if not primary_category:
        logger.debug("[overture_places_tool] No filters applied, using base SQL")
        return "select * from overture_places"

    sql = "select * from overture_places\nwhere "

    conditions = []
    if primary_category:
        primary_category_list_str = ", ".join([f"'{c}'" for c in primary_category])
        conditions.append(f"primary_category IN ({primary_category_list_str})")

    sql += " AND ".join(conditions)
    logger.debug("[overture_places_tool] Generated filtered SQL")

    return sql


def get_overture_places_tool(
    aoi_layer_id: str, feature_name: Optional[str] = None
) -> str:
    """Retrieve place point data from the Overture Maps Foundation dataset.

    This tool provides access to comprehensive points of interest data with intelligent
    semantic filtering capabilities. It can understand natural language requests for
    specific place types and convert them to appropriate database queries.

    The tool integrates multiple AI services:
    - RAG service for understanding available place primary categories
    - Gemini LLM for semantic interpretation of place type requests
    - Spatial filtering based on Area of Interest polygons
    - MotherDuck database for high-performance geospatial queries

    Place Data Includes:
    - Point geometries representing place locations
    - Primary category (food_and_drink, retail, healthcare, education, etc.)
    - Place names, addresses, and contact information
    - Business hours, ratings, and other metadata
    - Additional attributes from Overture Maps schema

    Args:
        aoi_layer_id: Unique identifier for the Area of Interest layer to spatially
                     filter places. Must exist in the user's current session.
        feature_name: Optional natural language description of desired place types.
                     Examples: 'restaurants', 'hospitals', 'schools', 'banks'.
                     If None, returns all places within the AOI.

    Returns:
        JSON string containing the processed layer data with place points
        and attributes, filtered by the specified AOI and feature criteria.

    Raises:
        Exception: If database query fails, AOI layer is invalid, or processing errors occur.

    Example:
        >>> get_overture_places_tool('layer_123', 'restaurants and cafes')
        '{"layers": [{"type": "FeatureCollection", "features": [...]}]}'
    """
    logger.info(
        f"[overture_places_tool] Retrieving Overture places data for AOI: {aoi_layer_id}"
    )
    try:
        if not feature_name:
            sql = "select * from overture_places"
            logger.debug(
                "[overture_places_tool] No feature name provided, using default SQL query"
            )
        else:
            sql = get_sql(feature_name)
            logger.debug(
                f"[overture_places_tool] Generated SQL query for feature: {feature_name}"
            )

        max_limit = 500_000_000

        sql = add_limit_to_sql(sql, max_limit)

        layers_data = handle_sql_tool_with_aoi(sql, aoi_layer_id)
        logger.info(
            f"[overture_places_tool] Successfully retrieved Overture places data: {len(layers_data)} layer(s)"
        )
        return layers_data
    except Exception as e:
        logger.error(
            f"[overture_places_tool] Failed to retrieve Overture places data: {e}",
            exc_info=True,
        )
        raise
