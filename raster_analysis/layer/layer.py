from typing import Any, Dict


class LayerInfo:
    def __init__(self, name_type: str):
        self.name_type: str = name_type
        parts = name_type.split("__")

        if len(parts) != 2:
            raise ValueError(
                f"Layer name `{name_type}` is invalid, should consist of layer name and unit separated by `__`"
            )

        if parts[0] == "is":
            self.data_type, self.name = parts
        else:
            self.name, self.data_type = parts


class DataLakeLayerInfoManager:
    DEFAULT_KEY = "_"

    def __init__(self):
        self._data_lake_info: Dict[str, Dict[str, Any]] = self._populate_data()

    def get_latest_version(self, layer_name: str) -> str:
        self._check_layer(layer_name)
        return self._data_lake_info[layer_name]["latest"]

    def get_pixel_value(self, layer_name: str, layer_value: Any) -> Any:
        inverted_map = self._get_inverted_value_map(layer_name)
        return self._get_value(layer_value, inverted_map)

    def get_layer_value(self, layer_name: str, pixel_value: Any) -> Any:
        value_map = self._get_value_map(layer_name)
        return self._get_value(pixel_value, value_map)

    def has_default_value(self, layer_name: str, no_data_value: Any):
        """
        Checks if the NoData is defined in a value map for the layer. This is used to determine if the NoData
        value should be filtered.
        """
        value_map = self._get_value_map(layer_name)
        return value_map and self.DEFAULT_KEY in value_map

    def _get_value_map(self, layer_name: str) -> Dict[Any, Any]:
        self._check_layer(layer_name)
        return self._data_lake_info[layer_name].get("value_map", {})

    def _get_inverted_value_map(self, layer_name: str) -> Dict[Any, Any]:
        value_map = self._get_value_map(layer_name)
        return {val: key for key, val in value_map.items()}

    def _check_layer(self, layer_name: str) -> None:
        if layer_name not in self._data_lake_info:
            raise ValueError(f"Layer {layer_name} not available in data API.")

    def _get_value(self, value: Any, value_map: Dict[Any, Any]):
        if value_map:
            if value in value_map:
                return value_map[value]
            elif self.DEFAULT_KEY in value_map:
                return value_map[self.DEFAULT_KEY]
            else:
                return None
        else:
            return value

    @staticmethod
    def _populate_data() -> Dict[str, Dict[str, Any]]:
        """
        Hardcoded data on latest alerts and value mappings. This will be replaced by call to
        data API when it's available.
        :return:
        """
        return {
            "umd_tree_cover_loss": {"latest": "v1.8"},
            "umd_regional_primary_forest_2001": {"latest": "v201901"},
            "umd_tree_cover_density_2000": {
                "latest": "v1.6",
                "value_map": {
                    "1": "10",
                    "2": "15",
                    "3": "20",
                    "4": "25",
                    "5": "30",
                    "6": "50",
                    "7": "75",
                },
            },
            "umd_tree_cover_density_2010": {
                "latest": "v1.6",
                "value_map": {
                    "1": "10",
                    "2": "15",
                    "3": "20",
                    "4": "25",
                    "5": "30",
                    "6": "50",
                    "7": "75",
                },
            },
            "umd_tree_cover_gain": {"latest": "v1.6"},
            "whrc_aboveground_biomass_stock_2000": {"latest": "v4"},
            "tsc_tree_cover_loss_drivers": {
                "latest": "v2020",
                "value_map": {
                    "1": "Commodity driven deforestation",
                    "2": "Shifting agriculture",
                    "3": "Forestry",
                    "4": "Wildfire",
                    "5": "Urbanization",
                    "_": "Unknown",
                },
            },
            "gfw_plantations": {
                "latest": "v1.3",
                "value_map": {
                    "1": "Fruit",
                    "2": "Fruit Mix",
                    "3": "Oil Palm ",
                    "4": "Oil Palm Mix",
                    "5": "Other",
                    "6": "Rubber",
                    "7": "Rubber Mix",
                    "8": "Unknown",
                    "9": "Unknown Mix",
                    "10": "Wood fiber / Timber",
                    "11": "Wood fiber / Timber Mix",
                },
            },
            "wdpa_protected_areas": {
                "latest": "v202007",
                "value_map": {"1": "Category Ia/b or II", "2": "Other Category"},
            },
            "esa_land_cover_2015": {
                "latest": "v20160111",
                "value_map": {
                    "10": "Agriculture",
                    "11": "Agriculture",
                    "12": "Agriculture",
                    "20": "Agriculture",
                    "30": "Agriculture",
                    "40": "Agriculture",
                    "50": "Forest",
                    "60": "Forest",
                    "61": "Forest",
                    "62": "Forest",
                    "70": "Forest",
                    "72": "Forest",
                    "80": "Forest",
                    "81": "Forest",
                    "82": "Forest",
                    "90": "Forest",
                    "100": "Forest",
                    "160": "Forest",
                    "170": "Forest",
                    "110": "Grassland",
                    "130": "Grassland",
                    "180": "Wetland",
                    "190": "Settlement",
                    "120": "Shrubland",
                    "121": "Shrubland",
                    "122": "Shrubland",
                    "140": "Sparse vegetation",
                    "150": "Sparse vegetation",
                    "151": "Sparse vegetation",
                    "152": "Sparse vegetation",
                    "153": "Sparse vegetation",
                    "200": "Bare",
                    "201": "Bare",
                    "202": "Bare",
                    "210": "Water",
                    "220": "Permanent snow and ice",
                    "_": "Unknown",
                },
            },
            "birdlife_alliance_for_zero_extinction_sites": {"latest": "v20200725"},
            "gmw_mangroves_1996": {"latest": "v20180701"},
            "gmw_mangroves_2016": {"latest": "v20180701"},
            "ifl_intact_forest_landscapes": {"latest": "v20180628"},
            "gfw_tiger_landscapes": {"latest": "v201904"},
            "landmark_land_rights": {"latest": "v20191111"},
            "gfw_land_rights": {"latest": "v2016"},
            "birdlife_key_biodiversity_areas": {"latest": "v20191211"},
            "gfw_mining": {"latest": "v20190205"},
            "gfw_peatlands": {"latest": "v20190103"},
            "gfw_oil_palm": {"latest": "v20191031"},
            "gfw_wood_fiber": {"latest": "v20200725"},
            "gfw_resource_rights": {"latest": "v2015"},
            "gfw_managed_forests": {"latest": "v20190103"},
            "rspo_oil_palm": {
                "latest": "v20200114",
                "value_map": {"1": "Certified", "2": "Unknown", "3": "Not certified"},
            },
            "idn_forest_area": {
                "latest": "v201709",
                "value_map": {
                    "1001": "Protected Forest",
                    "1003": "Production Forest",
                    "1004": "Limited Production Forest",
                    "1005": "Converted Production Forest",
                    "1007": "Other Utilization Area",
                    "1": "Sanctuary Reserves/Nature Conservation Area",
                    "1002": "Sanctuary Reserves/Nature Conservation Area",
                    "10021": "Sanctuary Reserves/Nature Conservation Area",
                    "10022": "Sanctuary Reserves/Nature Conservation Area",
                    "10023": "Sanctuary Reserves/Nature Conservation Area",
                    "10024": "Sanctuary Reserves/Nature Conservation Area",
                    "10025": "Sanctuary Reserves/Nature Conservation Area",
                    "10026": "Sanctuary Reserves/Nature Conservation Area",
                    "100201": "Marine Protected Areas",
                    "100211": "Marine Protected Areas",
                    "100221": "Marine Protected Areas",
                    "100201": "Marine Protected Areas",
                    "100201": "Marine Protected Areas",
                },
            },
            "per_forest_concessions": {
                "latest": "v20161001",
                "value_map": {
                    "1": "Conservation",
                    "2": "Ecotourism",
                    "3": "Nontimber Forest Products (Nuts)",
                    "4": "Nontimber Forest Products (Shiringa)",
                    "5": "Reforestation",
                    "6": "Timber Concession",
                    "7": "Wildlife",
                },
            },
            "bra_biomes": {
                "latest": "v20150601",
                "value_map": {
                    "1": "Caatinga",
                    "2": "Cerrado",
                    "3": "Pantanal",
                    "4": "Pampa",
                    "5": "Amazônia",
                    "6": "Mata Atlântica",
                },
            },
        }
