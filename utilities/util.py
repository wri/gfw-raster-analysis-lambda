

def unpack_count_histogram(stats, value_offset=0):

    print '---------- HERE ---------------'

    output_dict = {}

    for key, pixel_count in stats.iteritems():
        ras1, area = key.split('::')

        ras1 = value_offset + int(float(ras1))
        area_ha = float(area) * float(pixel_count) / 10000.

        try:
            output_dict[ras1] += area_ha
        except KeyError:
            output_dict[ras1] = area_ha

    print output_dict

    return output_dict
