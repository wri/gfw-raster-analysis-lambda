import os
import sys
import json
from unittest import TestCase


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import handler


class TestLandcover(TestCase):

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
		result = handler.landcover(payload, None)

		return json.loads(result['body'])['data']['attributes']['landcover']

	def test_idn_landcover_results(self):

		response = self.run_analysis('idn-landcover')

		self.assertEqual(len(response), 12)

		body_of_water = [x['result'] for x in response if x['className'] == 'Body of water'][0]
		self.assertEqual(body_of_water, 0)
