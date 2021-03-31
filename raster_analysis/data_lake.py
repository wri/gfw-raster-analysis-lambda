from datetime import date, datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict

from pandas import Series

from raster_analysis.globals import CO2_FACTOR
from raster_analysis.layer import Layer


def glad_date_decoder(s: Series) -> Dict[str, Series]:
    days_since_2015 = s % 10000
    ordinal_dates = days_since_2015 + date(2014, 12, 31).toordinal()
    str_dates = ordinal_dates.apply(
        lambda val: date.fromordinal(val).strftime("%Y-%m-%d")
    )
    return {"umd_glad_landsat_alerts__date": str_dates}


def glad_date_encoder(val: Any) -> List[Any]:
    as_date = datetime.strptime(val, "%Y-%m-%d")
    days_since_2015 = as_date.toordinal() - date(2014, 12, 31).toordinal()
    return [days_since_2015]


def glad_isoweek_decoder(s: Series):
    days_since_2015 = s % 10000
    ordinal_dates = days_since_2015 + date(2014, 12, 31).toordinal()
    dates = [date.fromordinal(ordinal) for ordinal in ordinal_dates]
    iso_week_dates = [(d - timedelta(days=d.isoweekday() - 1)) for d in dates]

    iso_weeks = list(map(lambda val: val.isocalendar()[0], iso_week_dates))
    years = list(map(lambda val: val.isocalendar()[1], iso_week_dates))

    return {
        "umd_glad_landsat_alerts__isoweek": iso_weeks,
        "umd_glad_landsat_alerts__year": years,
    }


# TODO refactor this when you start consuming from data API, using a dict here gets messy
LAYERS: Dict[str, Layer] = {
    "area__ha": Layer(layer="area__ha", version="virtual"),
    "alert__count": Layer(layer="alert__count", version="virtual"),
    "latitude": Layer(layer="latitude", version="virtual"),
    "longitude": Layer(layer="longitude", version="virtual"),
    "umd_tree_cover_loss__year": Layer(
        layer="umd_tree_cover_loss__year",
        version="v1.8",
        decoder=(lambda s: {"umd_tree_cover_loss__year": s + 2000}),
        encoder=(lambda val: [val - 2000]),
    ),
    # deprecated
    "umd_glad_alerts__date": Layer(
        layer="umd_glad_landsat_alerts__date",
        version="v1.7",
        decoder=glad_date_decoder,
        encoder=glad_date_encoder,
    ),
    # deprecated
    "umd_glad_alerts__isoweek": Layer(
        layer="umd_glad_landsat_alerts__date",
        version="v1.7",
        decoder=glad_isoweek_decoder,
    ),
    "umd_glad_landsat_alerts__date": Layer(
        layer="umd_glad_landsat_alerts__date",
        version="v1.7",
        decoder=glad_date_decoder,
        encoder=glad_date_encoder,
        is_conf_encoded=True,
    ),
    "umd_glad_landsat_alerts__isoweek": Layer(
        layer="umd_glad_landsat_alerts__date",
        version="v1.7",
        decoder=glad_isoweek_decoder,
    ),
    "umd_glad_landsat_alerts__confidence": Layer.from_encoding(
        "umd_glad_landsat_alerts__date",
        "v1.7",
        encoding={2: "", 3: "high"},
        alias="umd_glad_landsat_alerts__confidence",
    ),
    "is__umd_regional_primary_forest_2001": Layer(
        layer="is__umd_regional_primary_forest_2001", version="v201901"
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
        decoder=(lambda s: {"whrc_aboveground_co2_emissions__Mg": s * CO2_FACTOR}),
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
    "gfw_plantations__threshold": Layer.from_encoding(
        "gfw_plantations__threshold",
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
    "wdpa_protected_areas__threshold": Layer.from_encoding(
        "wdpa_protected_areas__threshold",
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
    "birdlife_alliance_for_zero_extinction_sites": Layer(
        layer="birdlife_alliance_for_zero_extinction_sites", version="v20200725"
    ),
    "gmw_mangroves_1996": Layer(layer="gmw_mangroves_1996", version="v20180701"),
    "gmw_mangroves_2016": Layer(layer="gmw_mangroves_2016", version="v20180701"),
    "ifl_intact_forest_landscapes": Layer(
        layer="ifl_intact_forest_landscapes", version="v20180628"
    ),
    "gfw_tiger_landscapes": Layer(layer="gfw_tiger_landscapes", version="v201904"),
    "landmark_land_rights": Layer(layer="landmark_land_rights", version="v20191111"),
    "gfw_land_rights": Layer(layer="gfw_land_rights", version="v2016"),
    "birdlife_key_biodiversity_areas": Layer(
        layer="birdlife_key_biodiversity_areas", version="v20191211"
    ),
    "gfw_mining": Layer(layer="gfw_mining", version="v20190205"),
    "gfw_peatlands": Layer(layer="gfw_peatlands", version="v20190103"),
    "gfw_oil_palm": Layer(layer="gfw_oil_palm", version="v20191031"),
    "gfw_wood_fiber": Layer(layer="gfw_wood_fiber", version="v20200725"),
    "gfw_resource_rights": Layer(layer="gfw_resource_rights", version="v2015"),
    "gfw_managed_forests": Layer(layer="gfw_managed_forests", version="v20190103"),
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
}
