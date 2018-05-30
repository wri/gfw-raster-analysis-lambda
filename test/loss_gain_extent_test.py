import os
import sys
import json
from unittest import TestCase


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import handler


class TestLossGainExtent(TestCase):

    def generate_payload(self, analysis_type='loss'):

        with open(os.path.join(root_dir, 'test/data/confresa.geojson')) as src_geojson:
            aoi = json.load(src_geojson)

        payload = {
         'body': json.dumps({'geojson': aoi}),
         'queryStringParameters': {'analysis': analysis_type, 'thresh': '30'}
        }

        return payload


    def run_analysis(self, analysis_type):


        payload = self.generate_payload(analysis_type)

        analysis_raster = 'test/data/10S_060W_{}.tif'.format(analysis_type)
        area_raster = 'test/data/10S_060W_area.tif'

        result = handler.analysis(payload, None, analysis_raster, area_raster)

        return json.loads(result['body'])



    def test_gain_results(self):

        response = self.run_analysis('gain')
        self.assertEqual(int(response['gain']), 401)

    def test_loss_results(self):

        response = self.run_analysis('loss')
        self.assertEqual(int(response['loss']), 17771)

    def test_extent_results(self):

        response = self.run_analysis('extent')
        self.assertEqual(int(response['extent']), 57735)

    def test_bad_thresh(self):
        payload = self.generate_payload()
        thresh = 15
        payload['queryStringParameters']['thresh'] = thresh

        response = handler.umd_loss_gain(payload, None)
        valid_thresh = [10, 30, 90]
        thresh_str = ', '.join([str(x) for x in valid_thresh])
        msg = 'thresh {} supplied, for this S3 endpoint must be one of {}'.format(thresh, thresh_str)
        response = json.loads(response['body'])['error']
        self.assertEqual(response, msg)
