from typing import Any, Dict


class LayerInfo:
    def __init__(self, name_type: str):
        self.name_type = name_type
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
    def __init__(self):
        self._data_lake_info = self._populate_data()

    def get_latest_version(self, layer_name: str) -> str:
        self._check_layer(layer_name)
        return self._data_lake_info[layer_name]["latest"]

    def get_pixel_value(self, layer_name: str, layer_value: Any) -> Any:
        inverted_map = self._get_inverted_value_map(layer_name)
        return self._get_value(layer_name, layer_value, inverted_map)

    def get_layer_value(self, layer_name: str, pixel_value: Any) -> Any:
        value_map = self._get_value_map(layer_name)
        return self._get_value(layer_name, pixel_value, value_map)

    def has_default_value(self, layer_name: str, no_data_value: Any):
        """
        Checks if the NoData is defined in a value map for the layer. This is used to determine if the NoData
        value should be filtered.
        """
        value_map = self._get_value_map(layer_name)
        return value_map and no_data_value in value_map

    def _get_value_map(self, layer_name: str) -> Dict[Any, Any]:
        self._check_layer(layer_name)
        return self._data_lake_info[layer_name].get("value_map", {})

    def _get_inverted_value_map(self, layer_name: str) -> Dict[Any, Any]:
        value_map = self._get_value_map(layer_name)
        return {val: key for key, val in value_map.items()}

    def _check_layer(self, layer_name: str) -> None:
        if layer_name not in self._data_lake_info:
            raise ValueError(f"Layer {layer_name} not available in data API.")

    @staticmethod
    def _get_value(layer_name: str, value: Any, value_map: Dict[Any, Any]):
        if value_map:
            if value in value_map:
                return value_map[value]
            else:
                raise KeyError(f"Value '{value}' not defined for layer '{layer_name}'")

    @staticmethod
    def _populate_data() -> Dict[str, Dict[str, Any]]:
        return {
            "umd_tree_cover_loss": {"latest": "v1.7"},
            "umd_regional_primary_forest_2001": {"latest": "v201901"},
            "umd_tree_cover_density_2000": {
                "latest": "v1.6",
                "value_map": {1: 10, 2: 15, 3: 20, 4: 25, 5: 30, 6: 50, 7: 75},
            },
            "umd_tree_cover_density_2010": {
                "latest": "v1.6",
                "value_map": {
                    1: "10",
                    2: "15",
                    3: "20",
                    4: "25",
                    5: "30",
                    6: "50",
                    7: "75",
                },
            },
            "umd_tree_cover_gain": {"latest": "v1.6"},
            "whrc_aboveground_biomass_stock_2000": {"latest": "v4"},
            "tsc_tree_cover_loss_drivers": {
                "latest": "v2020",
                "value_map": {
                    0: "Unknown",
                    1: "Commodity driven deforestation",
                    2: "Shifting agriculture",
                    3: "Forestry",
                    4: "Wildfire",
                    5: "Urbanization",
                },
            },
        }
