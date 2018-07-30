import os


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, 'data')


lulc_dict = {'primary-forest': os.path.join(data_dir, 'primary_forest.vrt'),
             'gfw-landcover-2015': os.path.join(data_dir, 'gfw_landcover.vrt'),
             'sea-landcover': os.path.join(data_dir, 'sea_landcover.vrt'),
             'idn-landcover': os.path.join(data_dir, 'idn_landcover.vrt')}

def ras_lkp(layer_name):
    return lulc_dict[layer_name]


def get_valid_layers():
    return lulc_dict.keys()


def build_lulc_lookup(layer_name):

    all_lulc_dict = {

    'gfw-landcover-2015':  {
        1: 'Agriculture',
        2: 'Forest',
        3: 'Grassland',
        4: 'Wetland',
        5: 'Settlement',
        6: 'Shrubland',
        7: 'Sparse vegetation',
        8: 'Bare',
        9: 'Water',
        10: 'Permanent snow and ice'
        },

    'primary-forest': {
        0: 'Not Primary Forest',
        1: 'Primary Forest'
        },

    'idn-landcover': {
        0: 'Secondary forest',
        1: 'Primary forest',
        2: 'Timber plantation',
        3: 'Agriculture',
        4: 'Settlement',
        5: 'Swamp',
        6: 'Grassland/shrub',
        7: 'Bare land',
        8: 'Estate crop plantation',
        9: 'Body of water',
        10: 'Fish pond',
        11: 'Mining'
    },

     'sea-landcover': {
        0: 'Mining',
        1: 'Mixed tree crops',
        2: 'No data',
        3: 'Oil palm plantation',
        4: 'Settlements',
        5: 'Swamp',
        6: 'Timber plantation',
        7: 'Primary forest',
        8: 'Water bodies',
        9: 'Bare land',
        10: 'Coastal fish pond',
        11: 'Rubber plantation',
        12: 'Agriculture',
        13: 'Secondary forest',
        14: r'Grassland/shrub'
    }
    }
               
    return all_lulc_dict[layer_name]









