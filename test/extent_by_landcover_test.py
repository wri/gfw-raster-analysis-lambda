import os
import sys
import json
from unittest import TestCase


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import handler


class TestExtentByLandCover(TestCase):

    def generate_payload(self, lulc_layer):

        with open(os.path.join(root_dir, 'test/data/id_950.geojson')) as src_geojson:
            aoi = json.load(src_geojson)

        payload = {
         'body': json.dumps({'geojson': aoi}),
         'queryStringParameters': {'layer': lulc_layer}
        }

        return payload

    def run_analysis(self, lulc_layer):

        payload = self.generate_payload(lulc_layer)

        kwargs = {
                    'lulc_raster': 'test/data/10N_100E_{}.tif'.format(lulc_layer),
                    'extent_raster': 'test/data/10N_100E_extent.tif',
                    'area_raster': 'test/data/10N_100E_area.tif'
                 }

        result = handler.extent_by_landcover(payload, None, **kwargs)

        return json.loads(result['body'])['data']['attributes']['landcover']

    def test_primary_forest_results(self):

        response = self.run_analysis('primary-forest')

        primary_forest = [x['result'] for x in response if x['className'] == 'Primary Forest'][0]
        not_primary_forest = [x['result'] for x in response if x['className'] == 'Not Primary Forest'][0]

        self.assertEqual(primary_forest, 59517.48647186706)
        self.assertEqual(not_primary_forest, 29187.073098955003)

