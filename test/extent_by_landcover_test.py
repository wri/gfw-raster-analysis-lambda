import os
import sys
import json
from unittest import TestCase


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import handler


class TestExtentByLandcover(TestCase):


    def generate_payload(self, location):

        with open(os.path.join(root_dir, 'test/data/palmoilmillgeneric.json')) as src_geojson:
            aoi_all = json.load(src_geojson)

        aoi = {
            'type': 'FeatureCollection',
            'features': [f for f in aoi_all['features'] if f['properties']['testing_id'] == location]
        }

        payload = {
         'body': json.dumps({'geojson': aoi}),
         'queryStringParameters': {'layer': 'primary-forest'}
        }

        return payload


    def run_analysis(self, location):

        payload = self.generate_payload(location)
        result = handler.extent_by_landcover(payload, None)

        return json.loads(result['body'])


    def get_location_list(self):
        with open(os.path.join(root_dir, 'test/data/palmoilmillgeneric.json')) as src_geojson:
            aoi = json.load(src_geojson)
        return [f['properties']['testing_id'] for f in aoi['features']]


    def get_validation_data(self):
        with open(os.path.join(root_dir, 'test/data/palmoilmillgeneric_primaryforest_treecover_QA.json')) as src:
            data = json.load(src)
        return data


    def extract_extent(self, response):
        result = [result for result in response['data']['attributes']['landcover'] if result['className'] == 'Primary Forest'][0]
        return int(result['result'])


    def validate(self, val1, val2):
        if isinstance(val1, list) and isinstance(val2, list):
            val1 = sum(val1)
            val2 = sum(val2)
        return abs(val1 - val2)/float(val2) <= 0.01


    def test_results_by_location(self):

        testing_ids = self.get_location_list()
        validation_data = self.get_validation_data()

        for testing_id in testing_ids:
            response = self.run_analysis(testing_id, 'primary-forest')
            self.assertEqual(int(self.validate(self.extract_extent(response), validation_data[testing_id]['extent'])), 1)
