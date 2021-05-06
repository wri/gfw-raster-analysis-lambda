from datetime import date, datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

from pandas import Series

from raster_analysis.globals import CO2_FACTOR
from raster_analysis.layer import Layer, Grid


def date_conf_decoder(layer: str, s: Series) -> Dict[str, Series]:
    days_since_2015 = s % 10000
    ordinal_dates = days_since_2015 + date(2014, 12, 31).toordinal()
    str_dates = ordinal_dates.apply(
        lambda val: date.fromordinal(val).strftime("%Y-%m-%d")
    )

    return {layer: str_dates}


def date_conf_encoder(val: Any) -> List[Any]:
    as_date = datetime.strptime(val, "%Y-%m-%d")
    days_since_2015 = as_date.toordinal() - date(2014, 12, 31).toordinal()
    return [days_since_2015]


def date_conf_isoweek_decoder(layer: str, s: Series):
    days_since_2015 = s % 10000
    ordinal_dates = days_since_2015 + date(2014, 12, 31).toordinal()
    dates = [date.fromordinal(ordinal) for ordinal in ordinal_dates]
    iso_week_dates = [(d - timedelta(days=d.isoweekday() - 1)) for d in dates]

    iso_weeks = list(map(lambda val: val.isocalendar()[1], iso_week_dates))
    years = list(map(lambda val: val.isocalendar()[0], iso_week_dates))

    base_name = layer.split("__")[0]
    return {f"{base_name}__isoweek": iso_weeks, f"{base_name}__year": years}


def year_decoder(layer, s):
    return {layer: s + 2000}


def year_encoder(val):
    return [val - 2000]


def co2_decoder(layer, s):
    return {"whrc_aboveground_co2_emissions__Mg": s * CO2_FACTOR}


# TODO refactor this when you start consuming from data API, using a dict here gets messy
LAYERS: Dict[str, Layer] = defaultdict(
    lambda: Layer(layer="count", version="virtual"),
    {
        "area__ha": Layer(layer="area__ha", version="virtual"),
        "alert__count": Layer(layer="alert__count", version="virtual"),
        "latitude": Layer(layer="latitude", version="virtual"),
        "longitude": Layer(layer="longitude", version="virtual"),
        "umd_tree_cover_loss__year": Layer(
            layer="umd_tree_cover_loss__year",
            version="v1.8",
            decoder=year_decoder,
            encoder=year_encoder,
        ),
        # deprecated
        "umd_glad_alerts__date": Layer(
            layer="umd_glad_landsat_alerts__date",
            version="v1.7",
            decoder=date_conf_decoder,
            encoder=date_conf_encoder,
        ),
        # deprecated
        "umd_glad_alerts__isoweek": Layer(
            layer="umd_glad_landsat_alerts__date",
            version="v1.7",
            decoder=date_conf_isoweek_decoder,
        ),
        "umd_glad_landsat_alerts__date": Layer(
            layer="umd_glad_landsat_alerts__date",
            version="v1.7",
            decoder=date_conf_decoder,
            encoder=date_conf_encoder,
            is_conf_encoded=True,
        ),
        "umd_glad_landsat_alerts__isoweek": Layer(
            layer="umd_glad_landsat_alerts__date",
            version="v1.7",
            decoder=date_conf_isoweek_decoder,
        ),
        "umd_glad_landsat_alerts__confidence": Layer.from_encoding(
            "umd_glad_landsat_alerts__date",
            "v1.7",
            encoding={2: "", 3: "high"},
            alias="umd_glad_landsat_alerts__confidence",
        ),
        "gfw_radd_alerts__date": Layer(
            layer="gfw_radd_alerts__date_conf",
            alias="gfw_radd_alerts__date",
            version="v20210328",
            decoder=date_conf_decoder,
            encoder=date_conf_encoder,
            grid=Grid(degrees=10, pixels=100000, tile_degrees=0.5),
        ),
        "gfw_radd_alerts__confidence": Layer.from_encoding(
            "gfw_radd_alerts__date_conf",
            "v20210328",
            encoding={2: "", 3: "high"},
            grid=Grid(degrees=10, pixels=100000, tile_degrees=0.5),
            alias="gfw_radd_alerts__confidence",
        ),
        "umd_glad_sentinel2_alerts__date": Layer(
            layer="umd_glad_sentinel2_alerts__date_conf",
            alias="umd_glad_sentinel2_alerts__date",
            version="v20210406",
            decoder=date_conf_decoder,
            encoder=date_conf_encoder,
            grid=Grid(degrees=10, pixels=100000, tile_degrees=0.5),
        ),
        "umd_glad_sentinel2_alerts__confidence": Layer.from_encoding(
            "umd_glad_sentinel2_alerts__date_conf",
            "v20210406",
            encoding={2: "", 3: "high"},
            grid=Grid(degrees=10, pixels=100000, tile_degrees=0.5),
            alias="umd_glad_sentinel2_alerts__confidence",
        ),
        "is__umd_regional_primary_forest_2001": Layer.boolean(
            "is__umd_regional_primary_forest_2001", "v201901"
        ),
        "umd_tree_cover_density_2000__threshold": Layer.from_encoding(
            "umd_tree_cover_density_2000__threshold",
            "v1.6",
            encoding={1: 10, 2: 15, 3: 20, 4: 25, 5: 30, 6: 50, 7: 75},
        ),
        "umd_tree_cover_density_2010__threshold": Layer.from_encoding(
            "umd_tree_cover_density_2010__threshold",
            "v1.6",
            encoding={1: 10, 2: 15, 3: 20, 4: 25, 5: 30, 6: 50, 7: 75},
        ),
        "is__umd_tree_cover_gain": Layer.boolean("is__umd_tree_cover_gain", "v1.6"),
        "whrc_aboveground_biomass_stock_2000__Mg_ha-1": Layer(
            layer="whrc_aboveground_biomass_stock_2000__Mg_ha-1",
            version="v4",
            is_area_density=True,
        ),
        "whrc_aboveground_co2_emissions__Mg": Layer(
            layer="whrc_aboveground_biomass_stock_2000__Mg_ha-1",
            version="v4",
            is_area_density=True,
            decoder=co2_decoder,
        ),
        "tsc_tree_cover_loss_drivers__type": Layer.from_encoding(
            "tsc_tree_cover_loss_drivers__type",
            "v2020",
            encoding=defaultdict(
                lambda: "Unknown",
                {
                    1: "Commodity driven deforestation",
                    2: "Shifting agriculture",
                    3: "Forestry",
                    4: "Wildfire",
                    5: "Urbanization",
                },
            ),
        ),
        "gfw_plantations__type": Layer.from_encoding(
            "gfw_plantations__type",
            "v1.3",
            encoding={
                1: "Fruit",
                2: "Fruit Mix",
                3: "Oil Palm ",
                4: "Oil Palm Mix",
                5: "Other",
                6: "Rubber",
                7: "Rubber Mix",
                8: "Unknown",
                9: "Unknown Mix",
                10: "Wood fiber / Timber",
                11: "Wood fiber / Timber Mix",
            },
        ),
        "wdpa_protected_areas__iucn_cat": Layer.from_encoding(
            "wdpa_protected_areas__iucn_cat",
            "v202007",
            encoding={1: "Category Ia/b or II", 2: "Other Category"},
        ),
        "esa_land_cover_2015__class": Layer.from_encoding(
            "esa_land_cover_2015__class",
            "v20160111",
            encoding=defaultdict(
                lambda: "Unknown",
                {
                    10: "Agriculture",
                    11: "Agriculture",
                    12: "Agriculture",
                    20: "Agriculture",
                    30: "Agriculture",
                    40: "Agriculture",
                    50: "Forest",
                    60: "Forest",
                    61: "Forest",
                    62: "Forest",
                    70: "Forest",
                    72: "Forest",
                    80: "Forest",
                    81: "Forest",
                    82: "Forest",
                    90: "Forest",
                    100: "Forest",
                    160: "Forest",
                    170: "Forest",
                    110: "Grassland",
                    130: "Grassland",
                    180: "Wetland",
                    190: "Settlement",
                    120: "Shrubland",
                    121: "Shrubland",
                    122: "Shrubland",
                    140: "Sparse vegetation",
                    150: "Sparse vegetation",
                    151: "Sparse vegetation",
                    152: "Sparse vegetation",
                    153: "Sparse vegetation",
                    200: "Bare",
                    201: "Bare",
                    202: "Bare",
                    210: "Water",
                    220: "Permanent snow and ice",
                },
            ),
        ),
        "is__birdlife_alliance_for_zero_extinction_sites": Layer.boolean(
            "is__birdlife_alliance_for_zero_extinction_sites", "v20200725"
        ),
        "is__gmw_mangroves_1996": Layer.boolean("is__gmw_mangroves_1996", "v20180701"),
        "is__gmw_mangroves_2016": Layer.boolean("is__gmw_mangroves_2016", "v20180701"),
        "ifl_intact_forest_landscapes__year": Layer(
            layer="ifl_intact_forest_landscapes__year", version="v20180628"
        ),
        "is__gfw_tiger_landscapes": Layer.boolean(
            "is__gfw_tiger_landscapes", "v201904"
        ),
        "is__landmark_land_rights": Layer.boolean(
            "is__landmark_land_rights", "v20191111"
        ),
        "is__gfw_land_rights": Layer.boolean("is__gfw_land_rights", "v2016"),
        "is__birdlife_key_biodiversity_areas": Layer.boolean(
            "is__birdlife_key_biodiversity_areas", "v20191211"
        ),
        "is__gfw_mining": Layer.boolean("is__gfw_mining", "v20190205"),
        "is__gfw_peatlands": Layer.boolean("is__gfw_peatlands", "v20190103"),
        "is__gfw_oil_palm": Layer.boolean("is__gfw_oil_palm", "v20191031"),
        "is__gfw_wood_fiber": Layer.boolean("is__gfw_wood_fiber", "v20200725"),
        "is__gfw_resource_rights": Layer.boolean("is__gfw_resource_rights", "v2015"),
        "is__gfw_managed_forests": Layer.boolean(
            "is__gfw_managed_forests", "v20190103"
        ),
        "rspo_oil_palm__certification_status": Layer(
            layer="rspo_oil_palm__certification_status",
            version="v20200114",
            encoding={1: "Certified", 2: "Unknown", 3: "Not certified"},
        ),
        "idn_forest_area__type": Layer.from_encoding(
            "idn_forest_area__type",
            "v201709",
            encoding={
                1001: "Protected Forest",
                1003: "Production Forest",
                1004: "Limited Production Forest",
                1005: "Converted Production Forest",
                1007: "Other Utilization Area",
                1: "Sanctuary Reserves/Nature Conservation Area",
                1002: "Sanctuary Reserves/Nature Conservation Area",
                10021: "Sanctuary Reserves/Nature Conservation Area",
                10022: "Sanctuary Reserves/Nature Conservation Area",
                10023: "Sanctuary Reserves/Nature Conservation Area",
                10024: "Sanctuary Reserves/Nature Conservation Area",
                10025: "Sanctuary Reserves/Nature Conservation Area",
                10026: "Sanctuary Reserves/Nature Conservation Area",
                100201: "Marine Protected Areas",
                100211: "Marine Protected Areas",
                100221: "Marine Protected Areas",
                100201: "Marine Protected Areas",
                100201: "Marine Protected Areas",
            },
        ),
        "per_forest_concession__type": Layer.from_encoding(
            "per_forest_concession__type",
            "v20161001",
            encoding={
                1: "Conservation",
                2: "Ecotourism",
                3: "Nontimber Forest Products (Nuts)",
                4: "Nontimber Forest Products (Shiringa)",
                5: "Reforestation",
                6: "Timber Concession",
                7: "Wildlife",
            },
        ),
        "bra_biome__name": Layer.from_encoding(
            "bra_biome__name",
            "v20150601",
            encoding={
                1: "Caatinga",
                2: "Cerrado",
                3: "Pantanal",
                4: "Pampa",
                5: "Amazônia",
                6: "Mata Atlântica",
            },
        ),
    },
)
