POLYGON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://geojson.org/schema/Polygon.json",
    "title": "GeoJSON Polygon",
    "type": "object",
    "required": ["type", "coordinates"],
    "properties": {
        "type": {"type": "string", "enum": ["Polygon"]},
        "coordinates": {
            "type": "array",
            "items": {
                "type": "array",
                "minItems": 4,
                "items": {"type": "array", "minItems": 2, "items": {"type": "number"}},
            },
        },
        "bbox": {"type": "array", "minItems": 4, "items": {"type": "number"}},
    },
}

MUTLIPOLYGON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://geojson.org/schema/MultiPolygon.json",
    "title": "GeoJSON MultiPolygon",
    "type": "object",
    "required": ["type", "coordinates"],
    "properties": {
        "type": {"type": "string", "enum": ["MultiPolygon"]},
        "coordinates": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "array",
                    "minItems": 4,
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "number"},
                    },
                },
            },
        },
        "bbox": {"type": "array", "minItems": 4, "items": {"type": "number"}},
    },
}

SCHEMA = {
    "type": "object",
    "properties": {
        "analysis_raster_id": {"type": "string"},
        "geometry": {"oneOf": [POLYGON_SCHEMA, MUTLIPOLYGON_SCHEMA]},
        "contextual_raster_ids": {"type": "array", "items": {"type": "string"}},
        "aggregate_raster_ids": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "object", "properties": {"name": {"type": "string"}}},
                    {"type": "object", "properties": {"price": {"type": "number"}}},
                ]
            },
        },
        "analyses": {
            "type": "array",
            "items": {"type": "string"},
            # {
            #     "anyOf": [
            #         {
            #             "type": "object",
            #             "properties": {"type": {"type": "string", "enum": ["count"]}},
            #         },
            #         {
            #             "type": "object",
            #             "properties": {"type": {"type": "string", "enum": ["area"]}},
            #         },
            #         {
            #             "type": "object",
            #             "properties": {"type": {"type": "string", "enum": ["sum"]}},
            #         },
            #     ]
            # },
        },
    },
    "required": ["analysis_raster_id", "geometry"],
}
