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


    def validate(self, response, validation_data, testing_id):
        # extract extent from response
        result = [result for result in response['data']['attributes']['landcover'] if result['className'] == 'Primary Forest']
        returned_extent = result[0]['result'] if result else 0

        # extract extent from validation data
        validation_extent = validation_data[testing_id]['extent'] if testing_id in validation_data.keys() else 0

        return abs(returned_extent - validation_extent)/(float(validation_extent + 1e-8)) * 100 <= 1.0


    def test_results_by_location(self):

        testing_ids = self.get_location_list()
        validation_data = self.get_validation_data()

        for testing_id in testing_ids:
            response = self.run_analysis(testing_id)
            self.assertEqual(int(self.validate(response, validation_data, testing_id)), 1)
