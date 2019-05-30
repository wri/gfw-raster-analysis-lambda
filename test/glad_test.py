# import os
# import sys
# import json
#
# from unittest import TestCase
#
# from api import app
#
# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#
# # set environment variable so that we know we're testing
# os.environ['ENV'] = 'test'
#
#
# class TestAlerts(TestCase):
#
#     # run the expensive raster analysis once, then check all outputs
#     # https://stackoverflow.com/a/38332812/4355916
#     @classmethod
#     def setUpClass(cls):
#
#         # create the test client, so we can pass POSTs directly to app
#         cls.app = app.test_client()
#
#         # load Bumba AOI
#         with open(os.path.join(root_dir, 'test/data/bumba.geojson')) as src_geojson:
#             aoi = json.load(src_geojson)
#
#         payload = json.dumps({'geojson': aoi})
#         params = {'aggregate_values': 'true',
#                   'aggregate_by': 'all',
#                   'period': '2015-05-23,2017-10-05'}
#
#         response = cls.app.post('glad-alerts', data=payload, query_string=params, content_type='application/json')
#
#         response = response.get_json()
#         cls.response_dict = response['data']['attributes']['value']['all']
#
#     def test_year_results(self):
#
#         year_list = self.response_dict['year']
#         self.assertEqual(len(year_list), 3)
#
#         alert_count = filter_rows(year_list, 2015)
#         self.assertEqual(alert_count, 9474)
#
#     def test_month_results(self):
#
#         month_list = self.response_dict['month']
#         self.assertEqual(len(month_list), 27)
#
#         alert_count = filter_rows(month_list, 2017, 9)
#         self.assertEqual(alert_count, 260)
#
#     def test_quarter_results(self):
#
#         quarter_list = self.response_dict['quarter']
#         self.assertEqual(len(quarter_list), 10)
#
#         alert_count = filter_rows(quarter_list, 2015, 2)
#         self.assertEqual(alert_count, 6891)
#
#     def test_week_results(self):
#
#         week_list = self.response_dict['week']
#         self.assertEqual(len(week_list), 61)
#
#         alert_count = filter_rows(week_list, 2016, 17)
#         self.assertEqual(alert_count, 244)
#
#     def test_day_results(self):
#
#         day_list = self.response_dict['day']
#         self.assertEqual(len(day_list), 61)
#
#         alert_count = filter_rows(day_list, 2016, '2016-03-10')
#
#         # how/why is this value so large? in the data but still . . . crazy
#         self.assertEqual(alert_count, 24913)
#
#     def test_total_results(self):
#         total_val = self.response_dict['total']
#         self.assertEqual(total_val, 86767)
#
#
# class TestAlertsConfOnly(TestCase):
#
#     # run the expensive raster analysis once, then check all outputs
#     # https://stackoverflow.com/a/38332812/4355916
#     @classmethod
#     def setUpClass(cls):
#
#         # create the test client, so we can pass POSTs directly to app
#         cls.app = app.test_client()
#
#         # load Bumba AOI
#         with open(os.path.join(root_dir, 'test/data/bumba.geojson')) as src_geojson:
#             aoi = json.load(src_geojson)
#
#         payload = json.dumps({'geojson': aoi})
#         params = {'aggregate_values': 'true',
#                   'aggregate_by': 'all',
#                   'gladConfirmOnly': 'TRUE',
#                   'period': '2015-10-01,2018-02-01'}
#
#         response = cls.app.post('glad-alerts', data=payload, query_string=params, content_type='application/json')
#
#         response = response.get_json()
#         cls.response_dict = response['data']['attributes']['value']['all']
#
#     def test_year_results(self):
#
#         year_list = self.response_dict['year']
#         self.assertEqual(len(year_list), 4)
#
#         alert_count = filter_rows(year_list, 2018)
#         self.assertEqual(alert_count, 3674)
#
#     def test_month_results(self):
#
#         month_list = self.response_dict['month']
#         self.assertEqual(len(month_list), 26)
#
#         alert_count = filter_rows(month_list, 2017, 12)
#         self.assertEqual(alert_count, 4)
#
#     def test_quarter_results(self):
#
#         quarter_list = self.response_dict['quarter']
#         self.assertEqual(len(quarter_list), 10)
#
#         alert_count = filter_rows(quarter_list, 2017, 4)
#         self.assertEqual(alert_count, 59)
#
#     def test_week_results(self):
#
#         week_list = self.response_dict['week']
#         self.assertEqual(len(week_list), 62)
#
#         alert_count = filter_rows(week_list, 2018, 3)
#         self.assertEqual(alert_count, 648)
#
#     def test_day_results(self):
#
#         day_list = self.response_dict['day']
#         self.assertEqual(len(day_list), 62)
#
#         alert_count = filter_rows(day_list, 2017, '2017-12-02')
#         self.assertEqual(alert_count, 4)
#
#     def test_total_results(self):
#         total_val = self.response_dict['total']
#         self.assertEqual(total_val, 82965)
#
#
# def filter_rows(row_list, year_val, time_val=None):
#
#     # first filter for our year of interest
#     filtered_rows = [x for x in row_list if x['year'] == year_val]
#
#     # if a time val specified in addition to year, use this to filter as well
#     if time_val:
#         row_keys = row_list[0].keys()
#         skip_list = ['year', 'count']
#         time_type = [x for x in row_keys if x not in skip_list][0]
#
#         filtered_rows = [x for x in filtered_rows if x[time_type] == time_val]
#
#     # grab the first row in this 1 object list, and return count
#     return filtered_rows[0]['count']
#
