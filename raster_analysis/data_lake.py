from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Callable
from collections import defaultdict

from numpy import ndarray
from pydantic import BaseModel
from pandas import Series

from raster_analysis.globals import CO2_FACTOR
from raster_analysis.query import Filter, Operator


class Layer:
    def __init__(
            self,
            layer: str,
            version: str,
            encoding: Dict[Any, Any] = {},
            value_decoder: Callable[[Series], Dict[str, Series]] = None,
            filter_encoder: Callable[[Filter], List[Filter]] = None,
            is_area_density: bool = False,
    ):
        self.layer: str = layer
        self.version: str = version
        self.is_area_density: bool = is_area_density
        if encoding:
            self.value_decoder = (lambda s: {layer: s.map(encoding)})
            self.filter_encoder = Layer._filter_encoder
        else:
            self.value_decoder: Callable[[Series], Series] = value_decoder
            self.filter_encoder: Callable[[Any], Any] = filter_encoder

    def __hash__(self):
        return hash(self.layer)

    @staticmethod
    def _default_filter_encoder(f):
        if f.layer.encoding:
            encoded_values = [dec_val for dec_val in f.layer.encoding.values() if f.value == dec_val]
        else:
            return [f.value]

        return [
            Filter(operator=f.op, layer=f.layer, value=enc_val)
            for enc_val in encoded_values
        ]


class GladDateFilter(Filter):
    def apply_filter(self, window: ndarray) -> ndarray:
        # remove confidence before applying date filter
        window_days = window % 10000
        super().apply_filter(window_days)


def glad_date_value_decoder(s: Series) -> Dict[str, Series]:
    days_since_2015 = s % 10000
    ordinal_dates = days_since_2015 + date(2014, 12, 31).toordinal()
    str_dates = ordinal_dates.apply(
        lambda val: date.fromordinal(val).strftime("%Y-%m-%d")
    )
    return {"umd_glad_alerts__date": str_dates}


def glad_date_filter_encoder(f: Filter) -> List[Filter]:
    as_date = datetime.strptime(f.value, "%Y-%m-%d")
    days_since_2015 = as_date.toordinal() - date(2014, 12, 31).toordinal()
    return [GladDateFilter(op=f.op, layer=f.layer, value=days_since_2015)]


def glad_isoweek_value_decoder(s: Series):
    days_since_2015 = s % 10000
    ordinal_dates = days_since_2015 + date(2014, 12, 31).toordinal()
    dates = [date.fromordinal(ordinal) for ordinal in ordinal_dates]
    iso_week_dates = [
        (d - timedelta(days=d.isoweekday() - 1)) for d in dates
    ]

    iso_weeks = iso_week_dates.apply(lambda val: val.isocalendar()[0])
    years = iso_week_dates.apply(lambda val: val.isocalendar()[1])

    return {
        "umd_glad_alerts__isoweek": iso_weeks,
        "umd_glad_alerts__year": years
    }

# def glad_conf_status_decoder(s):
#     return int((s - (s % 10000)) / 10000)
#

LAYERS: Dict[str, Layer] = {
    "umd_tree_cover_loss__year": Layer(
        "umd_tree_cover_loss__year",
        "v1.8",
        value_decoder=(lambda s: {"umd_tree_cover_loss__year": s + 2000}),
        filter_encoder=(lambda f: [Filter(op=f.op, layer=f.layer, value=f.value - 2000)]),
    ),
    "umd_glad_alerts__date": Layer(
        "umd_glad_alerts__date",
        "v1.7",
        value_decoder=glad_date_value_decoder,
        filter_encoder=glad_date_filter_encoder,
    ),
    "umd_glad_alerts__isoweek": Layer(
        "umd_glad_alerts__date",
        "v1.7",
        value_decoder=glad_isoweek_value_decoder,
    ),
    "is__umd_regional_primary_forest_2001": Layer("is__umd_regional_primary_forest_2001", "v201901"),
    "umd_tree_cover_density_2000__threshold": Layer("umd_tree_cover_density_2000__threshold", "v1.6", encoder=(lambda s: s.map({
        1: 10,
        2: 15,
        3: 20,
        4: 25,
        5: 30,
        6: 50,
        7: 75,

    }))),
    "umd_tree_cover_density_2000__threshold": Layer("umd_tree_cover_density_2000__threshold", "v1.6", encoding={
        1: 10,
        2: 15,
        3: 20,
        4: 25,
        5: 30,
        6: 50,
        7: 75,
    }),
    "umd_tree_cover_gain_year": Layer("umd_tree_cover_gain_year", "v1.6"),
    "whrc_aboveground_biomass_stock_2000__Mg_ha-1": Layer("whrc_aboveground_biomass_stock_2000__Mg_ha-1", "v4", is_area_density=True),
    "whrc_aboveground_co2_emissions__Mg": Layer(
        "whrc_aboveground_biomass_stock_2000__Mg_ha-1",
        "v4",
        is_area_density=True,
        value_decoder=(lambda s: {"whrc_aboveground_co2_emissions__Mg": s * CO2_FACTOR})
    ),
    "tsc_tree_cover_loss_drivers__type": Layer("tsc_tree_cover_loss_drivers__type", "v2020", encoding=defaultdict(
        lambda: "Unknown",
        {
            1: "Commodity driven deforestation",
            2: "Shifting agriculture",
            3: "Forestry",
            4: "Wildfire",
            5: "Urbanization",
        }
    )),
    "gfw_plantations__threshold": Layer("gfw_plantations__threshold", "v1.3", encoding={
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
    }),
    "wdpa_protected_areas__threshold": Layer("wdpa_protected_areas__threshold", "v202007", encoding={
        1: "Category Ia/b or II",
        2: "Other Category"
    }),
    "esa_land_cover_2015__class": Layer("esa_land_cover_2015__class", "v20160111", encoding=defaultdict(
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
        })),
    "birdlife_alliance_for_zero_extinction_sites": Layer("birdlife_alliance_for_zero_extinction_sites", "v20200725"),
    "gmw_mangroves_1996": Layer("gmw_mangroves_1996", "v20180701"),
    "gmw_mangroves_2016": Layer("gmw_mangroves_2016", "v20180701"),
    "ifl_intact_forest_landscapes": Layer("ifl_intact_forest_landscapes", "v20180628"),
    "gfw_tiger_landscapes": Layer("gfw_tiger_landscapes", "v201904"),
    "landmark_land_rights": Layer("landmark_land_rights", "v20191111"),
    "gfw_land_rights": Layer("gfw_land_rights", "v2016"),
    "birdlife_key_biodiversity_areas": Layer("birdlife_key_biodiversity_areas", "v20191211"),
    "gfw_mining": Layer("gfw_mining", "v20190205"),
    "gfw_peatlands": Layer("gfw_peatlands", "v20190103"),
    "gfw_oil_palm": Layer("gfw_oil_palm", "v20191031"),
    "gfw_wood_fiber": Layer("gfw_wood_fiber", "v20200725"),
    "gfw_resource_rights": Layer("gfw_resource_rights", "v2015"),
    "gfw_managed_forests": Layer("gfw_managed_forests", "v20190103"),
    "rspo_oil_palm__certification_status": Layer("rspo_oil_palm__certification_status", "v20200114", encoding={1: "Certified", 2: "Unknown", 3: "Not certified"}),
    "idn_forest_area__type": Layer("idn_forest_area__type", "v201709", encoding={
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
    }),
    "per_forest_concession__type": Layer("per_forest_concession__type", "v20161001", encoding={
        1: "Conservation",
        2: "Ecotourism",
        3: "Nontimber Forest Products (Nuts)",
        4: "Nontimber Forest Products (Shiringa)",
        5: "Reforestation",
        6: "Timber Concession",
        7: "Wildlife",
    }),
    "bra_biome__name": Layer("bra_biome__name", "v20150601", encoding={
        1: "Caatinga",
        2: "Cerrado",
        3: "Pantanal",
        4: "Pampa",
        5: "Amazônia",
        6: "Mata Atlântica",
    })
}