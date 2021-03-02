from typing import Any, Dict, List
from collections import defaultdict

from pydantic import BaseModel

from raster_analysis.query import LayerInfo

class DataLakeLayerInfo(BaseModel):
    version: str
    encoding: defaultdict[Any, Any] = defaultdict()

    def encode(self, val: Any) -> List[Any]:
        """
        Get list of encoded values that map to a particular decoded value
        """
        if self.encoding:
            return [dec_val for dec_val in self.encoding.values() if val == dec_val]
        else:
            return [val]


class DataLakeLayerInfoManager(BaseModel):
    layers: Dict[LayerInfo, DataLakeLayerInfo]

    def __init__(self):
        self.layers: DataLakeLayerInfo = self._populate_data()

    @staticmethod
    def _populate_data() -> Dict[LayerInfo, DataLakeLayerInfo]:
        """
        Hardcoded data on latest alerts and value mappings. This will be replaced by call to
        data API when it's available.
        :return:
        """
        return {
            LayerInfo("umd_tree_cover_loss__year"): DataLakeLayerInfo(version="v1.8"),
            LayerInfo("is__umd_regional_primary_forest_2001"):  DataLakeLayerInfo(version="v201901"),
            LayerInfo("umd_tree_cover_density_2000__threshold"): DataLakeLayerInfo(version="v1.6", encoding={
                1: 10,
                2: 15,
                3: 20,
                4: 25,
                5: 30,
                6: 50,
                7: 75,
            }),
           LayerInfo("umd_tree_cover_density_2000__threshold"): DataLakeLayerInfo(version="v1.6", encoding={
               1: 10,
               2: 15,
               3: 20,
               4: 25,
               5: 30,
               6: 50,
               7: 75,
           }),
            LayerInfo("umd_tree_cover_gain_year"): DataLakeLayerInfo(version="v1.6"),
            LayerInfo("whrc_aboveground_biomass_stock_2000__Mg_ha-1"): DataLakeLayerInfo(version="v4"),
            LayerInfo("tsc_tree_cover_loss_drivers__type"): DataLakeLayerInfo(version="v2020", encoding=defaultdict(
                lambda: "Unknown",
                {
                    1: "Commodity driven deforestation",
                    2: "Shifting agriculture",
                    3: "Forestry",
                    4: "Wildfire",
                    5: "Urbanization",
                }
            )),
            LayerInfo("gfw_plantations__threshold"): DataLakeLayerInfo(version="v1.3", encoding={
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
            LayerInfo("wdpa_protected_areas__threshold"): DataLakeLayerInfo(version="v202007", encoding={
                1: "Category Ia/b or II",
                2: "Other Category"
            }),
            LayerInfo("esa_land_cover_2015__class"): DataLakeLayerInfo(version="v20160111", encoding=defaultdict(
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
            LayerInfo("birdlife_alliance_for_zero_extinction_sites"): DataLakeLayerInfo(version="v20200725"),
            LayerInfo("gmw_mangroves_1996"): DataLakeLayerInfo(version="v20180701"),
            LayerInfo("gmw_mangroves_2016"): DataLakeLayerInfo(version="v20180701"),
            LayerInfo("ifl_intact_forest_landscapes"): DataLakeLayerInfo(version="v20180628"),
            LayerInfo("gfw_tiger_landscapes"): DataLakeLayerInfo(version="v201904"),
            LayerInfo("landmark_land_rights"): DataLakeLayerInfo("v20191111"),
            LayerInfo("gfw_land_rights"): DataLakeLayerInfo(version="v2016"),
            LayerInfo("birdlife_key_biodiversity_areas"): DataLakeLayerInfo(version="v20191211"),
            LayerInfo("gfw_mining"): DataLakeLayerInfo(version="v20190205"),
            LayerInfo("gfw_peatlands"): DataLakeLayerInfo(version="v20190103"),
            LayerInfo("gfw_oil_palm"): DataLakeLayerInfo(version="v20191031"),
            LayerInfo("gfw_wood_fiber"): DataLakeLayerInfo(version="v20200725"),
            LayerInfo("gfw_resource_rights"): DataLakeLayerInfo(version="v2015"),
            LayerInfo("gfw_managed_forests"): DataLakeLayerInfo(version="v20190103"),
            LayerInfo("rspo_oil_palm__certification_status"): DataLakeLayerInfo(version="v20200114", encoding={1: "Certified", 2: "Unknown", 3: "Not certified"}),
            LayerInfo("idn_forest_area__type"): DataLakeLayerInfo(version="v201709", encoding={
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
            LayerInfo("per_forest_concession__type"): DataLakeLayerInfo(version="v20161001", encoding={
                1: "Conservation",
                2: "Ecotourism",
                3: "Nontimber Forest Products (Nuts)",
                4: "Nontimber Forest Products (Shiringa)",
                5: "Reforestation",
                6: "Timber Concession",
                7: "Wildlife",
            }),
            LayerInfo("bra_biome__name"): DataLakeLayerInfo(version="v20150601", encoding={
                1: "Caatinga",
                2: "Cerrado",
                3: "Pantanal",
                4: "Pampa",
                5: "Amazônia",
                6: "Mata Atlântica",
            })
        }

