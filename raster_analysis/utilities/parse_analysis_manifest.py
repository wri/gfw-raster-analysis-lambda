import json
from jsonschema import validate

def parse_analysis_manifest(manifest_path, schema_path):
    """"
    Parse an analysis manifest and return lists as variables
    """
    with open(manifest_path, 'r') as f:
        d = json.load(f)
    with open(schema_path, 'r') as f:
        s = json.load(f)
    # Check manifest is valid
    validate(instance=d, schema=s)
    # Parse parameters as lists
    analysis_id = d['id']
    raster_paths = [p['path'] for p in d['tilesets']]
    raster_ids = [p['id'] for p in d['tilesets']]
    raster_nodata_values = [p['no_data_value'] for p in d['tilesets']]
    agg_stats = d['aggregate_stats']
    return analysis_id, raster_ids, raster_paths, raster_nodata_values, agg_stats
