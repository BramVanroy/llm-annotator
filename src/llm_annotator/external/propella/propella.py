import json
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, List, Type, Union, cast

from pydantic import BaseModel, ConfigDict, Field


SYSTEM_PROMPT = """Annotate the document. Any language; assess quality within its linguistic norms. Respond with a JSON object:
content_integrity: technical completeness (complete|mostly_complete|fragment|severely_degraded)
content_ratio: content vs navigation/boilerplate ratio (complete_content|mostly_content|mixed_content|mostly_navigation|minimal_content)
content_length: substantive words (substantial 2k+|moderate 500-2k|brief 100-500|minimal <100)
one_sentence_description: neutral ~10 word summary in English
content_type[]: functional purpose (analytical|instructional|reference|procedural|qa_structured|conversational|creative|transactional|boilerplate|news_report|opinion_editorial|review_critique|technical_documentation|specification_standard|legal_document|press_release|structured_data|source_code)
business_sector[]: industry domain (academic_research|education_sector|technology_software|hardware_electronics|healthcare_medical|pharmaceutical_biotech|financial_services|legal_services|government_public|manufacturing_industrial|mining_resources|chemicals_materials|energy_utilities|retail_commerce|wholesale_distribution|real_estate_construction|transportation_logistics|automotive_industry|telecommunications|media_entertainment|advertising_marketing|hospitality_tourism|agriculture_food|environmental_services|aerospace_defense|insurance_industry|nonprofit_ngo|consulting_professional|human_resources|security_cyber|gaming_industry|gambling_betting|travel_aviation|food_beverage_hospitality|consumer_goods|general_interest|other)
technical_content[]: specialized knowledge (code_heavy|math_heavy|scientific|data_heavy|engineering|basic_technical|non_technical)
content_quality: writing/presentation quality (excellent|good|adequate|poor|unacceptable)
information_density: signal vs padding (dense|adequate|moderate|thin|empty)
educational_value: teaching potential (high|moderate|basic|minimal|none)
reasoning_indicators: logical analysis depth (analytical|explanatory|basic_reasoning|minimal|none)
audience_level: assumed background (expert|advanced|general|beginner|youth|children)
commercial_bias: promotional influence (none|minimal|moderate|heavy|pure_marketing)
time_sensitivity: temporal decay (evergreen|slowly_changing|regularly_updating|time_sensitive)
content_safety: harmful content (safe|mild_concerns|nsfw|harmful|illegal)
pii_presence: private individual data (no_pii|contains_pii)
regional_relevance[]: geographic/cultural context (european|north_american|east_asian|south_asian|southeast_asian|middle_eastern|sub_saharan_african|latin_american|oceanian|central_asian|russian_sphere|global|culturally_neutral|indeterminate)
country_relevance[]: specific countries as ISO names, or supranational|none
"""

USER_PROMPT = """<start_of_document>
{content}
<end_of_document>
"""

ANNOTATOR_SYSTEM_PROMPT = """You are an expert content analysis assistant specializing in document annotations for LLM pretraining data. Your team is curating a multilingual dataset for language model training. Your task is to analyze documents and annotate them with specific properties that will later on be used to filter the dataset. The user will provide a document inside of "<start_of_document>" and "<end_of_document>" tags. Analyze the content of the document systematically and objectively. Respond with your annotations in JSON format, following the annotation framework below.

# Annotation Framework
## Output Requirements
- You must respond with a JSON object that matches the specified schema.
- Use the exact enum values provided in the property descriptions.
- Ensure all fields are included.
- For multi-select properties, always return arrays (even if only one value applies). Multi-select fields: content_type, business_sector, technical_content, regional_relevance, country_relevance. All other properties are single-select strings.
- Do not include any explanatory text, comments, or additional formatting.

## Key Principles
* Objective assessment: Base decisions on clear criteria, not subjective preferences.
* Completeness: Address all properties for every document.
* Consistency: Apply the same standards across all documents.
* Multilinguality: The user provided document can be in any language, the language itself should not influence the annotations.

## Properties to Annotate
The annotation framework evaluates documents across 18 key properties organized into six main categories:

**Core Content Properties:**
- Content Integrity: Completeness and technical quality (complete, mostly_complete, fragment, severely_degraded)
- Content Ratio: Proportion of meaningful content vs navigation/UI elements (complete_content, mostly_content, mixed_content, mostly_navigation, minimal_content)
- Content Length: Amount of substantive content (substantial, moderate, brief, minimal)

**Content Classification:**
- One-Sentence Description: Ultra-short neutral description; exactly one sentence; target 8-15 words (soft max 20)
- Content Type: Functional structure and purpose (analytical, instructional, reference, procedural, qa_structured, conversational, creative, transactional, boilerplate, news_report, opinion_editorial, review_critique, technical_documentation, specification_standard, legal_document, press_release, structured_data, source_code)
- Business Sector: Industry domain relevance (see Detailed Property Descriptions for exact enum values)
- Technical Content: Type and intensity of specialized knowledge (code_heavy, math_heavy, scientific, data_heavy, engineering, basic_technical, non_technical)

**Quality and Value Assessment:**
- Content Quality: Overall writing and presentation quality (excellent, good, adequate, poor, unacceptable)
- Information Density: Ratio of valuable information to redundancy (dense, adequate, moderate, thin, empty)
- Educational Value: Potential for teaching and learning (high, moderate, basic, minimal, none)
- Reasoning Indicators: Presence of logical reasoning and analysis (analytical, explanatory, basic_reasoning, minimal, none)

**Audience and Purpose:**
- Audience Level: Target sophistication level (expert, advanced, general, beginner, youth, children)
- Commercial Bias: Commercial influence on objectivity (none, minimal, moderate, heavy, pure_marketing)
- Time-Sensitivity: How content value changes over time (evergreen, slowly_changing, regularly_updating, time_sensitive)

**Safety and Compliance:**
- Content Safety: Presence of inappropriate or harmful content (safe, mild_concerns, nsfw, harmful, illegal)
- PII Presence: Contains personally identifiable information (no_pii, contains_pii)

**Geographic Relevance:**
- Regional Relevance: Primary regional context (european, north_american, east_asian, south_asian, southeast_asian, middle_eastern, sub_saharan_african, latin_american, oceanian, central_asian, russian_sphere, global, culturally_neutral, indeterminate)
- Country Relevance: Specific country relevance (array of country names or special values: "supranational", "none")


{property_descriptions}

## JSON Schema for the Response
Return a single JSON object that strictly conforms to the following JSON Schema:
```json
{json_schema}
```

## Multilingual Annotation Guidelines

### Universal Principles
1. **Evaluate content quality within language context** - Don't penalize non-English content for being non-English
2. **Consider linguistic norms** - Writing styles, sentence lengths, and paragraph structures vary by language
3. **Respect script directionality** - RTL languages (Arabic, Hebrew) may have different navigation patterns
4. **Account for morphological complexity** - Agglutinative/polysynthetic languages pack more information per word

### Language-Specific Considerations

Here are some examples of language-specific considerations:

**Chinese/Japanese:**
- Character count more relevant than word count
- Lack of spaces between words is normal
- Mixed script usage (especially Japanese) is standard

**Arabic/Hebrew/Persian:**
- RTL text direction affects layout assessment
- Diacritical marks may be absent in informal content
- Mixed Arabic/English is common in technical content

**Indian Languages (Hindi, Bengali, Tamil, etc.):**
- Code-mixing with English is extremely common and acceptable
- Technical terms often borrowed from English
- Multiple scripts may appear in same document

**European Languages:**
- Formal/informal distinctions (tu/vous, du/Sie) indicate audience
- Compound words affect word count metrics
- Regional variants (Brazilian vs European Portuguese, Spanish vs Catalan, etc.) are both valid

# Annotation Workflow
- The user will provide a document in "<start_of_document>" and "<end_of_document>" tags. Analyze the content of the document systematically and objectively
- You must respond with a valid JSON object that matches the schema above.
- Use the exact enum values provided in the property descriptions
- Ensure all required fields are included
- For multi-select properties, always return arrays, even if only one value applies (content_type, business_sector, technical_content, regional_relevance, country_relevance). All other properties are single-select strings.
- Do not include any explanatory text, comments, or formatting
"""

ANNOTATOR_USER_PROMPT = """Analyze the following document and provide annotations in JSON format according to the annotation framework. Return only the JSON object.
<start_of_document>
{content}
<end_of_document>"""


# Default max length for one_sentence_description field
ONE_SENTENCE_DESCRIPTION_MAX_LENGTH = 200


class ContentIntegrity(str, Enum):
    """Content completeness and technical quality"""

    COMPLETE = "complete"
    MOSTLY_COMPLETE = "mostly_complete"
    FRAGMENT = "fragment"
    SEVERELY_DEGRADED = "severely_degraded"


class ContentRatio(str, Enum):
    """Ratio of meaningful content vs navigation/UI elements"""

    COMPLETE_CONTENT = "complete_content"
    MOSTLY_CONTENT = "mostly_content"
    MIXED_CONTENT = "mixed_content"
    MOSTLY_NAVIGATION = "mostly_navigation"
    MINIMAL_CONTENT = "minimal_content"


class ContentLength(str, Enum):
    """Amount of substantive content"""

    SUBSTANTIAL = "substantial"  # 500+ words
    MODERATE = "moderate"  # 100-500 words
    BRIEF = "brief"  # 20-100 words
    MINIMAL = "minimal"  # <20 words


class ContentType(str, Enum):
    """Primary purpose and type of content"""

    ANALYTICAL = "analytical"
    INSTRUCTIONAL = "instructional"
    REFERENCE = "reference"
    PROCEDURAL = "procedural"
    QA_STRUCTURED = "qa_structured"
    CONVERSATIONAL = "conversational"
    CREATIVE = "creative"
    TRANSACTIONAL = "transactional"
    BOILERPLATE = "boilerplate"
    NEWS_REPORT = "news_report"
    OPINION_EDITORIAL = "opinion_editorial"
    REVIEW_CRITIQUE = "review_critique"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    SPECIFICATION_STANDARD = "specification_standard"
    LEGAL_DOCUMENT = "legal_document"
    PRESS_RELEASE = "press_release"
    STRUCTURED_DATA = "structured_data"
    SOURCE_CODE = "source_code"


class BusinessSector(str, Enum):
    """Industry domain(s) for sector classification (multi-select)"""

    ACADEMIC_RESEARCH = "academic_research"
    EDUCATION_SECTOR = "education_sector"
    TECHNOLOGY_SOFTWARE = "technology_software"
    HARDWARE_ELECTRONICS = "hardware_electronics"
    HEALTHCARE_MEDICAL = "healthcare_medical"
    PHARMACEUTICAL_BIOTECH = "pharmaceutical_biotech"
    FINANCIAL_SERVICES = "financial_services"
    LEGAL_SERVICES = "legal_services"
    GOVERNMENT_PUBLIC = "government_public"
    MANUFACTURING_INDUSTRIAL = "manufacturing_industrial"
    MINING_RESOURCES = "mining_resources"
    CHEMICALS_MATERIALS = "chemicals_materials"
    ENERGY_UTILITIES = "energy_utilities"
    RETAIL_COMMERCE = "retail_commerce"
    WHOLESALE_DISTRIBUTION = "wholesale_distribution"
    REAL_ESTATE_CONSTRUCTION = "real_estate_construction"
    TRANSPORTATION_LOGISTICS = "transportation_logistics"
    AUTOMOTIVE_INDUSTRY = "automotive_industry"
    TELECOMMUNICATIONS = "telecommunications"
    MEDIA_ENTERTAINMENT = "media_entertainment"
    ADVERTISING_MARKETING = "advertising_marketing"
    HOSPITALITY_TOURISM = "hospitality_tourism"
    AGRICULTURE_FOOD = "agriculture_food"
    ENVIRONMENTAL_SERVICES = "environmental_services"
    AEROSPACE_DEFENSE = "aerospace_defense"
    INSURANCE_INDUSTRY = "insurance_industry"
    NONPROFIT_NGO = "nonprofit_ngo"
    CONSULTING_PROFESSIONAL = "consulting_professional"
    HUMAN_RESOURCES = "human_resources"
    SECURITY_CYBER = "security_cyber"
    GAMING_INDUSTRY = "gaming_industry"
    GAMBLING_BETTING = "gambling_betting"
    TRAVEL_AVIATION = "travel_aviation"
    FOOD_BEVERAGE_HOSPITALITY = "food_beverage_hospitality"
    CONSUMER_GOODS = "consumer_goods"
    GENERAL_INTEREST = "general_interest"
    OTHER = "other"


class TechnicalContent(str, Enum):
    """Type and intensity of specialized technical knowledge"""

    CODE_HEAVY = "code_heavy"
    MATH_HEAVY = "math_heavy"
    SCIENTIFIC = "scientific"
    DATA_HEAVY = "data_heavy"
    ENGINEERING = "engineering"
    BASIC_TECHNICAL = "basic_technical"
    NON_TECHNICAL = "non_technical"


class InformationDensity(str, Enum):
    """Ratio of valuable information to redundancy and padding"""

    DENSE = "dense"
    ADEQUATE = "adequate"
    MODERATE = "moderate"
    THIN = "thin"
    EMPTY = "empty"


class ContentQuality(str, Enum):
    """Overall quality considering writing, value, and presentation"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class AudienceLevel(str, Enum):
    """Intended sophistication level and background knowledge assumptions"""

    EXPERT = "expert"
    ADVANCED = "advanced"
    GENERAL = "general"
    BEGINNER = "beginner"
    YOUTH = "youth"
    CHILDREN = "children"


class CommercialBias(str, Enum):
    """Commercial influence on objectivity and informational value"""

    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HEAVY = "heavy"
    PURE_MARKETING = "pure_marketing"


class ContentSafety(str, Enum):
    """Presence of inappropriate, harmful, or legally problematic content"""

    SAFE = "safe"
    MILD_CONCERNS = "mild_concerns"
    NSFW = "nsfw"
    HARMFUL = "harmful"
    ILLEGAL = "illegal"


class EducationalValue(str, Enum):
    """Potential for teaching, learning, and knowledge transfer"""

    HIGH = "high"
    MODERATE = "moderate"
    BASIC = "basic"
    MINIMAL = "minimal"
    NONE = "none"


class ReasoningIndicators(str, Enum):
    """Presence and quality of logical reasoning and analysis"""

    ANALYTICAL = "analytical"
    EXPLANATORY = "explanatory"
    BASIC_REASONING = "basic_reasoning"
    MINIMAL = "minimal"
    NONE = "none"


class RegionalRelevance(str, Enum):
    """Primary regional, cultural, or geopolitical sphere(s)"""

    EUROPEAN = "european"
    NORTH_AMERICAN = "north_american"
    EAST_ASIAN = "east_asian"
    SOUTH_ASIAN = "south_asian"
    SOUTHEAST_ASIAN = "southeast_asian"
    MIDDLE_EASTERN = "middle_eastern"
    SUB_SAHARAN_AFRICAN = "sub_saharan_african"
    LATIN_AMERICAN = "latin_american"
    OCEANIAN = "oceanian"
    CENTRAL_ASIAN = "central_asian"
    RUSSIAN_SPHERE = "russian_sphere"
    GLOBAL = "global"
    CULTURALLY_NEUTRAL = "culturally_neutral"
    INDETERMINATE = "indeterminate"


class TimeSensitivity(str, Enum):
    """How time-sensitive the content is"""

    EVERGREEN = "evergreen"
    SLOWLY_CHANGING = "slowly_changing"
    REGULARLY_UPDATING = "regularly_updating"
    TIME_SENSITIVE = "time_sensitive"


class PiiPresence(str, Enum):
    """Presence of personally identifiable information"""

    NO_PII = "no_pii"
    CONTAINS_PII = "contains_pii"


class Country(str, Enum):
    """
    Country names for country relevance classification.
    Based on ISO 3166-1 standard - the authoritative international standard
    for country codes maintained by the International Organization for Standardization.

    Includes all 249 entities from ISO 3166-1: 193 UN member states,
    2 UN observer states, plus dependent territories and special areas.

    References:
    - https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes
    - https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area
    """

    # UN Member States (193 total) and UN Observer States (2 total)
    AFGHANISTAN = "afghanistan"
    ALBANIA = "albania"
    ALGERIA = "algeria"
    ANDORRA = "andorra"
    ANGOLA = "angola"
    ANTIGUA_AND_BARBUDA = "antigua_and_barbuda"
    ARGENTINA = "argentina"
    ARMENIA = "armenia"
    AUSTRALIA = "australia"
    AUSTRIA = "austria"
    AZERBAIJAN = "azerbaijan"
    BAHAMAS = "bahamas"
    BAHRAIN = "bahrain"
    BANGLADESH = "bangladesh"
    BARBADOS = "barbados"
    BELARUS = "belarus"
    BELGIUM = "belgium"
    BELIZE = "belize"
    BENIN = "benin"
    BHUTAN = "bhutan"
    BOLIVIA = "bolivia"
    BOSNIA_AND_HERZEGOVINA = "bosnia_and_herzegovina"
    BOTSWANA = "botswana"
    BRAZIL = "brazil"
    BRUNEI = "brunei"
    BULGARIA = "bulgaria"
    BURKINA_FASO = "burkina_faso"
    BURUNDI = "burundi"
    CABO_VERDE = "cabo_verde"
    CAMBODIA = "cambodia"
    CAMEROON = "cameroon"
    CANADA = "canada"
    CENTRAL_AFRICAN_REPUBLIC = "central_african_republic"
    CHAD = "chad"
    CHILE = "chile"
    CHINA = "china"
    COLOMBIA = "colombia"
    COMOROS = "comoros"
    CONGO = "congo"
    CONGO_DEMOCRATIC_REPUBLIC = "congo_democratic_republic"
    COOK_ISLANDS = "cook_islands"
    COSTA_RICA = "costa_rica"
    CROATIA = "croatia"
    CUBA = "cuba"
    CYPRUS = "cyprus"
    CZECH_REPUBLIC = "czech_republic"
    DENMARK = "denmark"
    DJIBOUTI = "djibouti"
    DOMINICA = "dominica"
    DOMINICAN_REPUBLIC = "dominican_republic"
    ECUADOR = "ecuador"
    EGYPT = "egypt"
    EL_SALVADOR = "el_salvador"
    EQUATORIAL_GUINEA = "equatorial_guinea"
    ERITREA = "eritrea"
    ESTONIA = "estonia"
    ESWATINI = "eswatini"
    ETHIOPIA = "ethiopia"
    FIJI = "fiji"
    FINLAND = "finland"
    FRANCE = "france"
    GABON = "gabon"
    GAMBIA = "gambia"
    GEORGIA = "georgia"
    GERMANY = "germany"
    GHANA = "ghana"
    GREECE = "greece"
    GRENADA = "grenada"
    GUATEMALA = "guatemala"
    GUINEA = "guinea"
    GUINEA_BISSAU = "guinea_bissau"
    GUYANA = "guyana"
    HAITI = "haiti"
    HONDURAS = "honduras"
    HUNGARY = "hungary"
    ICELAND = "iceland"
    INDIA = "india"
    INDONESIA = "indonesia"
    IRAN = "iran"
    IRAQ = "iraq"
    IRELAND = "ireland"
    ISRAEL = "israel"
    ITALY = "italy"
    IVORY_COAST = "ivory_coast"
    JAMAICA = "jamaica"
    JAPAN = "japan"
    JORDAN = "jordan"
    KAZAKHSTAN = "kazakhstan"
    KENYA = "kenya"
    KIRIBATI = "kiribati"
    NORTH_KOREA = "north_korea"
    SOUTH_KOREA = "south_korea"
    KOSOVO = "kosovo"
    KUWAIT = "kuwait"
    KYRGYZSTAN = "kyrgyzstan"
    LAOS = "laos"
    LATVIA = "latvia"
    LEBANON = "lebanon"
    LESOTHO = "lesotho"
    LIBERIA = "liberia"
    LIBYA = "libya"
    LIECHTENSTEIN = "liechtenstein"
    LITHUANIA = "lithuania"
    LUXEMBOURG = "luxembourg"
    MADAGASCAR = "madagascar"
    MALAWI = "malawi"
    MALAYSIA = "malaysia"
    MALDIVES = "maldives"
    MALI = "mali"
    MALTA = "malta"
    MARSHALL_ISLANDS = "marshall_islands"
    MAURITANIA = "mauritania"
    MAURITIUS = "mauritius"
    MEXICO = "mexico"
    MICRONESIA = "micronesia"
    MOLDOVA = "moldova"
    MONACO = "monaco"
    MONGOLIA = "mongolia"
    MONTENEGRO = "montenegro"
    MOROCCO = "morocco"
    MOZAMBIQUE = "mozambique"
    MYANMAR = "myanmar"
    NAMIBIA = "namibia"
    NAURU = "nauru"
    NEPAL = "nepal"
    NETHERLANDS = "netherlands"
    NEW_ZEALAND = "new_zealand"
    NICARAGUA = "nicaragua"
    NIGER = "niger"
    NIGERIA = "nigeria"
    NIUE = "niue"
    NORTH_MACEDONIA = "north_macedonia"
    NORWAY = "norway"
    OMAN = "oman"
    PAKISTAN = "pakistan"
    PALAU = "palau"
    PALESTINE = "palestine"  # UN Observer State
    PANAMA = "panama"
    PAPUA_NEW_GUINEA = "papua_new_guinea"
    PARAGUAY = "paraguay"
    PERU = "peru"
    PHILIPPINES = "philippines"
    POLAND = "poland"
    PORTUGAL = "portugal"
    QATAR = "qatar"
    ROMANIA = "romania"
    RUSSIA = "russia"
    RWANDA = "rwanda"
    SAINT_KITTS_AND_NEVIS = "saint_kitts_and_nevis"
    SAINT_LUCIA = "saint_lucia"
    SAINT_VINCENT_AND_THE_GRENADINES = "saint_vincent_and_the_grenadines"
    SAMOA = "samoa"
    SAN_MARINO = "san_marino"
    SAO_TOME_AND_PRINCIPE = "sao_tome_and_principe"
    SAUDI_ARABIA = "saudi_arabia"
    SENEGAL = "senegal"
    SERBIA = "serbia"
    SEYCHELLES = "seychelles"
    SIERRA_LEONE = "sierra_leone"
    SINGAPORE = "singapore"
    SLOVAKIA = "slovakia"
    SLOVENIA = "slovenia"
    SOLOMON_ISLANDS = "solomon_islands"
    SOMALIA = "somalia"
    SOUTH_AFRICA = "south_africa"
    SOUTH_SUDAN = "south_sudan"
    SPAIN = "spain"
    SRI_LANKA = "sri_lanka"
    SUDAN = "sudan"
    SURINAME = "suriname"
    SWEDEN = "sweden"
    SWITZERLAND = "switzerland"
    SYRIA = "syria"
    TAJIKISTAN = "tajikistan"
    TANZANIA = "tanzania"
    THAILAND = "thailand"
    TIMOR_LESTE = "timor_leste"
    TOGO = "togo"
    TONGA = "tonga"
    TRINIDAD_AND_TOBAGO = "trinidad_and_tobago"
    TUNISIA = "tunisia"
    TURKEY = "turkey"
    TURKMENISTAN = "turkmenistan"
    TUVALU = "tuvalu"
    UGANDA = "uganda"
    UKRAINE = "ukraine"
    UNITED_ARAB_EMIRATES = "united_arab_emirates"
    UNITED_KINGDOM = "united_kingdom"
    UNITED_STATES = "united_states"
    URUGUAY = "uruguay"
    UZBEKISTAN = "uzbekistan"
    VANUATU = "vanuatu"
    VATICAN_CITY = "vatican_city"  # UN Observer State
    VENEZUELA = "venezuela"
    VIETNAM = "vietnam"
    YEMEN = "yemen"
    ZAMBIA = "zambia"
    ZIMBABWE = "zimbabwe"

    # # Dependent Territories and Special Administrative Regions (from ISO 3166-1)
    ALAND_ISLANDS = "aland_islands"  # Finland
    AMERICAN_SAMOA = "american_samoa"  # United States
    ANGUILLA = "anguilla"  # United Kingdom
    ANTARCTICA = "antarctica"  # Antarctic Treaty
    ARUBA = "aruba"  # Netherlands
    ASCENSION_ISLAND = "ascension_island"  # United Kingdom
    BERMUDA = "bermuda"  # United Kingdom
    BRITISH_VIRGIN_ISLANDS = "british_virgin_islands"  # United Kingdom
    CAYMAN_ISLANDS = "cayman_islands"  # United Kingdom
    CHRISTMAS_ISLAND = "christmas_island"  # Australia
    COCOS_ISLANDS = "cocos_islands"  # Australia
    CURACAO = "curacao"  # Netherlands
    FALKLAND_ISLANDS = "falkland_islands"  # United Kingdom
    FAROE_ISLANDS = "faroe_islands"  # Denmark
    FRENCH_GUIANA = "french_guiana"  # France
    FRENCH_POLYNESIA = "french_polynesia"  # France
    GIBRALTAR = "gibraltar"  # United Kingdom
    GREENLAND = "greenland"  # Denmark
    GUADELOUPE = "guadeloupe"  # France
    GUAM = "guam"  # United States
    GUERNSEY = "guernsey"  # United Kingdom
    HONG_KONG = "hong_kong"  # China
    ISLE_OF_MAN = "isle_of_man"  # United Kingdom
    JERSEY = "jersey"  # United Kingdom
    MACAU = "macau"  # China
    MARTINIQUE = "martinique"  # France
    MAYOTTE = "mayotte"  # France
    MONTSERRAT = "montserrat"  # United Kingdom
    NEW_CALEDONIA = "new_caledonia"  # France
    NORFOLK_ISLAND = "norfolk_island"  # Australia
    NORTHERN_MARIANA_ISLANDS = "northern_mariana_islands"  # United States
    PITCAIRN_ISLANDS = "pitcairn_islands"  # United Kingdom
    PUERTO_RICO = "puerto_rico"  # United States
    REUNION = "reunion"  # France
    SAINT_BARTHELEMY = "saint_barthelemy"  # France
    SAINT_HELENA = "saint_helena"  # United Kingdom
    SAINT_MARTIN = "saint_martin"  # France
    SAINT_PIERRE_AND_MIQUELON = "saint_pierre_and_miquelon"  # France
    SINT_MAARTEN = "sint_maarten"  # Netherlands
    SVALBARD_AND_JAN_MAYEN = "svalbard_and_jan_mayen"  # Norway
    TAIWAN = "taiwan"  # China (disputed)
    TOKELAU = "tokelau"  # New Zealand
    TRISTAN_DA_CUNHA = "tristan_da_cunha"  # United Kingdom
    TURKS_AND_CAICOS_ISLANDS = "turks_and_caicos_islands"  # United Kingdom
    US_VIRGIN_ISLANDS = "us_virgin_islands"  # United States
    WALLIS_AND_FUTUNA = "wallis_and_futuna"  # France
    WESTERN_SAHARA = "western_sahara"  # Disputed


class CountryRelevanceSpecial(str, Enum):
    """
    Special values for country relevance classification from annotation guidelines.
    These are used when content doesn't relate to specific countries.
    """

    SUPRANATIONAL = "supranational"
    NONE = "none"


def create_annotation_response_model(
    one_sentence_description_max_length: int = ONE_SENTENCE_DESCRIPTION_MAX_LENGTH,
) -> Type[BaseModel]:
    """
    Factory function to create an AnnotationResponse model with configurable max_length
    for the one_sentence_description field.

    Args:
        one_sentence_description_max_length: Maximum length for the one_sentence_description field.
                                             Defaults to ONE_SENTENCE_DESCRIPTION_MAX_LENGTH (200).

    Returns:
        A Pydantic model class with the specified configuration.
    """

    class _AnnotationResponse(BaseModel):
        """
        Property annotation pydantic model for LLM pretraining data.
        It captures all 18 properties as defined in the annotation guidelines for consistently identifying high-value content for language model training.
        """

        # Property 1: Content Integrity
        content_integrity: ContentIntegrity = Field(
            ...,
            description="Completeness and technical quality of the content itself",
        )

        # Property 2: Content Ratio
        content_ratio: ContentRatio = Field(
            ...,
            description="Ratio of meaningful content vs navigation/UI elements",
        )

        # Property 3: Content Length
        content_length: ContentLength = Field(
            ...,
            description="Amount of substantive content, ignoring navigation and boilerplate",
        )

        # Property 4: One-Sentence Description
        one_sentence_description: str = Field(
            ...,
            description="Ultra-short neutral description of the document. Exactly one sentence. Target 8-15 words (soft max 20). Neutral tone; avoid boilerplate intros and calls to action.",
            max_length=one_sentence_description_max_length,
        )

        # Property 5: Content Type (multi-select)
        content_type: List[ContentType] = Field(
            ...,
            description="Primary purpose and type of content - always return an array (one or more types)",
            min_length=1,
            max_length=5,
        )

        # Property 6: Business Sector (multi-select)
        business_sector: List[BusinessSector] = Field(
            ...,
            description="Industry sector(s) - always return an array (one or more sectors)",
            min_length=1,
            max_length=10,
        )

        # Property 7: Technical Content (multi-select)
        technical_content: List[TechnicalContent] = Field(
            ...,
            description="Type and intensity of specialized technical knowledge - always return an array (one or more types)",
            min_length=1,
        )

        # Property 8: Information Density
        information_density: InformationDensity = Field(
            ...,
            description="Ratio of valuable information to redundancy, padding, and repetition",
        )

        # Property 9: Content Quality
        content_quality: ContentQuality = Field(
            ...,
            description="Overall quality considering writing excellence, substantive value, and presentation",
        )

        # Property 10: Audience Level
        audience_level: AudienceLevel = Field(
            ...,
            description="Intended sophistication level and background knowledge assumptions",
        )

        # Property 11: Commercial Bias
        commercial_bias: CommercialBias = Field(
            ...,
            description="How much commercial interests influence objectivity and informational value",
        )

        # Property 12: Time Sensitivity
        time_sensitivity: TimeSensitivity = Field(
            ..., description="How time-sensitive the content is"
        )

        # Property 13: Content Safety
        content_safety: ContentSafety = Field(
            ...,
            description="Presence of inappropriate, harmful, or legally problematic content",
        )

        # Property 14: Educational Value
        educational_value: EducationalValue = Field(
            ...,
            description="Potential for teaching, learning, and knowledge transfer",
        )

        # Property 15: Reasoning Indicators
        reasoning_indicators: ReasoningIndicators = Field(
            ...,
            description="Presence and quality of logical reasoning, analysis, and explanatory content",
        )

        # Property 16: PII Presence
        pii_presence: PiiPresence = Field(
            ...,
            description="Whether the content contains personally identifiable information",
        )

        # Property 17: Regional Relevance (multi-select)
        regional_relevance: List[RegionalRelevance] = Field(
            ...,
            description="Primary regional, cultural, or geopolitical sphere(s) - always return an array (one or multiple regions)",
            min_length=1,
            max_length=3,
        )

        # Property 18: Country Relevance (multi-select)
        country_relevance: List[Union[Country, CountryRelevanceSpecial]] = (
            Field(
                ...,
                description="Specific country/countries the content mentions or is relevant for (or special values for supranational/non-country-specific) - always return an array (one or more countries/special values)",
                min_length=1,
                max_length=10,
            )
        )

        model_config = ConfigDict(
            validate_assignment=True,
            extra="forbid",  # Don't allow extra fields
            json_schema_extra={
                "example": {
                    "content_integrity": "complete",
                    "content_ratio": "mostly_content",
                    "content_length": "substantial",
                    "one_sentence_description": "API reference for payment endpoints and error codes.",
                    "content_type": ["analytical", "instructional"],
                    "business_sector": [
                        "academic_research",
                        "technology_software",
                    ],
                    "technical_content": ["scientific", "data_heavy"],
                    "information_density": "dense",
                    "content_quality": "excellent",
                    "audience_level": "expert",
                    "commercial_bias": "none",
                    "time_sensitivity": "slowly_changing",
                    "content_safety": "safe",
                    "educational_value": "high",
                    "reasoning_indicators": "analytical",
                    "pii_presence": "no_pii",
                    "regional_relevance": ["european"],
                    "country_relevance": ["germany"],
                }
            },
        )

    return _AnnotationResponse


def flatten_model_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all #/$defs/... references and remove $defs from a Pydantic JSON Schema.

    - Recursively resolves $ref entries that point into local $defs
    - Preserves sibling constraints next to $ref by shallow-merging into the resolved target
    - Drops any nested $defs occurrences
    """
    schema_copy = deepcopy(schema)
    defs = schema_copy.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node.get("$ref")
                extra = {
                    k: v
                    for k, v in node.items()
                    if k != "$ref" and k != "$defs"
                }
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    name = ref.split("/")[-1]
                    replacement = deepcopy(defs.get(name, {}))
                    resolved_replacement = resolve(replacement)
                    resolved_extra = resolve(extra)
                    if isinstance(resolved_replacement, dict) and isinstance(
                        resolved_extra, dict
                    ):
                        return {**resolved_replacement, **resolved_extra}
                    return resolved_replacement
                resolved_extra = resolve(extra)
                return {
                    **({"$ref": ref}),
                    **(
                        resolved_extra
                        if isinstance(resolved_extra, dict)
                        else {}
                    ),
                }
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return cast(dict[str, Any], resolve(schema_copy))


def get_annotation_response_schema(
    use_country_enum: bool = True,
    flatten: bool = True,
    as_string: bool = False,
    minify: bool = True,
    one_sentence_description_max_length=ONE_SENTENCE_DESCRIPTION_MAX_LENGTH,
    compact_whitespace: bool = True,
) -> Union[dict, str]:
    """
    Build the JSON Schema for `AnnotationResponse` with an option to avoid large country enums.

    - If `use_country_enum` is True (default), the schema uses enum definitions for
      `country_relevance` items as generated by Pydantic.
    - If `use_country_enum` is False, `country_relevance` becomes a list of strings
      (no enum) while the property's description still contains the full list of
      valid values. This avoids very large enum blocks for APIs that do not support them.
    - If `flatten` is True (default), inline all local $defs via `flatten_model_json_schema`.
    - If `as_string` is True, return the schema as a JSON string. When `as_string`
      is True and `minify` is True (default), emit compact JSON with no extra
      whitespace to reduce token usage. If `minify` is False, pretty-print with indentation.
    - If `compact_whitespace` is True (default), adds x-guidance directive to enforce
      compact JSON output with no tabs, newlines, or extra whitespace between tokens.
      This prevents models from generating whitespace-heavy malformed JSON.
    """
    schema = create_annotation_response_model(
        one_sentence_description_max_length
    ).model_json_schema()

    # Add x-guidance directive for llguidance to enforce compact JSON (no tabs/newlines/whitespace)
    if compact_whitespace:
        schema["x-guidance"] = {"whitespace_flexible": False}

    if not use_country_enum:
        # Construct the list of valid values from the enums but do not emit them as enum types
        valid_values = [e.value for e in Country] + [
            e.value for e in CountryRelevanceSpecial
        ]

        country_prop = schema.get("properties", {}).get("country_relevance")
        if isinstance(country_prop, dict):
            existing_description = country_prop.get("description", "")

            # Ensure the property is an array of strings without duplicating the long values list
            country_prop["type"] = "array"
            country_prop["items"] = {"type": "string"}

            # Retain minItems and other constraints already present on the property

            # Put the full list of valid values only in the property description (not in items)
            values_text = f" Valid values: {', '.join(valid_values)}"
            if (
                existing_description
                and "Valid values:" not in existing_description
            ):
                country_prop["description"] = (
                    existing_description.rstrip() + values_text
                )
            elif not existing_description:
                country_prop["description"] = values_text.strip()

    if flatten:
        schema = flatten_model_json_schema(schema)

    if as_string:
        if minify:
            return json.dumps(
                schema, separators=(",", ":"), ensure_ascii=False
            )
        return json.dumps(schema, indent=2, ensure_ascii=False)

    return schema


# Default AnnotationResponse model with default max_length
AnnotationResponse = create_annotation_response_model()


TRUNCATION_TAG = "<truncated_content>"


def truncate_content(content: str, max_content_chars: int) -> str:
    if max_content_chars > 0 and len(content) > max_content_chars:
        return f"{content[:max_content_chars]}\n{TRUNCATION_TAG}"
    return content


ROOT_PATH = Path(__file__).parent
property_descriptions = ROOT_PATH.joinpath(
    "property_descriptions.md"
).read_text(encoding="utf-8")


def create_messages(
    document_text: str, max_content_chars: int = 50_000
) -> list[dict]:
    document_text = truncate_content(document_text, max_content_chars)
    user_prompt = USER_PROMPT.format(content=document_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return messages


schema_str = get_annotation_response_schema(
    as_string=True, one_sentence_description_max_length=150
)
annotator_system_prompt = ANNOTATOR_SYSTEM_PROMPT.format(
    json_schema=schema_str, property_descriptions=property_descriptions
)


def create_annotator_messages(
    document_text: str, max_content_chars: int = 50_000
) -> list[dict]:
    document_text = truncate_content(document_text, max_content_chars)
    user_prompt = ANNOTATOR_USER_PROMPT.format(content=document_text)

    messages = [
        {"role": "system", "content": annotator_system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages
