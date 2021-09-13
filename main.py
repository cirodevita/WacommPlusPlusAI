import pandas as pd
import json
import netCDF4
from datetime import datetime, timedelta
from configparser import ConfigParser
from shapely.geometry import Point, Polygon

cfg = ConfigParser()
cfg.read('config.ini')


# LOAD AREAS INFO
def load_areas():
    with open(cfg.get('files', 'AREAS')) as input_file:
        data = json.load(input_file)
        areas = data["features"]

    return areas


# REMOVE MEASURES DUPLICATES
def remove_measures_duplicates():
    max_measures = []

    lines = pd.read_csv(cfg.get('files', 'MEASURES'), delimiter=';')
    for line in lines.iterrows():
        id = line[1]['NUMERO SCHEDA']
        date = line[1]['DATA PRELIEVO'] + cfg.get('variables', 'SAMPLE_HOUR')
        outcome = line[1]['ESITO']
        site = line[1]['SITO']
        lat = line[1]['LATITUDINE_DEF']
        lon = line[1]['LONGITUDINE_DEF']

        if not any(d['id'] == id for d in max_measures):
            max_measures.append({"id": id, 'site_name': site, 'datetime': date, 'latitude': lat, 'longitude': lon,
                                 'outcome': outcome})
        else:
            index = [i for i, _ in enumerate(max_measures) if _['id'] == id][0]
            if outcome > max_measures[index]['outcome']:
                max_measures[index]['outcome'] = outcome
                max_measures[index]['datetime'] = date
                max_measures[index]['latitude'] = lat
                max_measures[index]['longitude'] = lon

    return max_measures


def get_index_lat_long(lat, long, min_lat, max_lat, min_long, max_long):
    N = len(lat)
    sum_delta_lat = 0
    for i in range(1, N):
        sum_delta_lat += lat[i] - lat[i-1]
    delta_lat = sum_delta_lat/N

    M = len(long)
    sum_delta_long = 0
    for i in range(1, M):
        sum_delta_long += long[i] - long[i - 1]
    delta_long = sum_delta_long / M

    index_min_lat = round((min_lat - lat[0])/delta_lat)
    index_max_lat = round((max_lat - lat[0])/delta_lat)
    index_min_long = round((min_long - long[0])/delta_long)
    index_max_long = round((max_long - long[0])/delta_long)

    # index_min_lat1 = min(range(len(lat)), key=lambda i: abs(lat[i] - min_lat))
    # index_max_lat1 = min(range(len(lat)), key=lambda i: abs(lat[i] - max_lat))
    # index_min_long1 = min(range(len(long)), key=lambda i: abs(long[i] - min_long))
    # index_max_long1 = min(range(len(long)), key=lambda i: abs(long[i] - max_long))

    return index_min_lat, index_max_lat, index_min_long, index_max_long


# CREATE DATASET
def create_dataset(areas, max_measures):
    my_dataset = []
    f = open("url_mancanti.txt", "a")

    for measure in max_measures:
        index = [i for i, _ in enumerate(areas) if _['properties']['DENOMINAZI'] == measure['site_name']][0]

        bbox = areas[index]['bbox']
        min_long = bbox[0]
        min_lat = bbox[1]
        max_long = bbox[2]
        max_lat = bbox[3]

        coordinates = areas[index]['geometry']['coordinates'][0][0]

        area_poly = Polygon(coordinates)

        date = measure['datetime']

        sample = measure['outcome']

        _dataset = {"features": [], "label": sample, "site": measure['site_name']}
        features = []

        for i in range(cfg.getint('variables', 'START'), cfg.getint('variables', 'END') + 1):
            reference_hour = datetime.strptime(date, '%d/%m/%y/%H:%M') - timedelta(hours=i)
            formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

            year = str(reference_hour.year)
            month = f'{reference_hour.month:02}'
            day = f'{reference_hour.day:02}'

            url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + ".nc"

            try:
                dataset = netCDF4.Dataset(url)
                long = dataset.variables['longitude']
                lat = dataset.variables['latitude']

                index_min_lat, index_max_lat, index_min_long, index_max_long = get_index_lat_long(lat, long,
                                                                                                  min_lat, max_lat,
                                                                                                  min_long, max_long)

                concentration = dataset.variables['conc'][0]
                max = float("-inf")
                for k in range(0, 2):
                    for i in range(index_min_lat, index_max_lat+1):
                        for j in range(index_min_long, index_max_long+1):
                            point = Point(long[j], lat[i])
                            if point.within(area_poly):
                                if concentration[k][i][j] > max:
                                    max = concentration[k][i][j]
                features.append(str(max))
            except:
                f.write(url + '\n')
                features.append('--')

        _dataset["features"] = features

        my_dataset.append(_dataset)

    f.close()

    return my_dataset


if __name__ == "__main__":
    areas = load_areas()
    max_measures = remove_measures_duplicates()
    dataset = create_dataset(areas, max_measures)

    with open('dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
