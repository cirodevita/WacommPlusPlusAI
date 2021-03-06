import concurrent.futures
import math
from functools import partial
import multiprocessing
import numpy as np
import time
import pandas as pd
import statistics
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
def remove_measures_duplicates(areas):
    measures = []

    list_files = cfg.get('files', 'MEASURES')
    files = json.loads(list_files)

    list = cfg.get('variables', 'YEAR')
    years = json.loads(list)

    for file in files:
        lines = pd.read_csv(file, delimiter=';')
        for line in lines.iterrows():
            if line[1]['ANNO ACCETTAZIONE'] in years:
                date = str(line[1]['DATA PRELIEVO']) + cfg.get('variables', 'SAMPLE_HOUR')
                try:
                    outcome = float(line[1]['ESITO'])
                except:
                    outcome = line[1]['ESITO']
                year = str(line[1]['ANNO ACCETTAZIONE'])
                try:
                    code = str(line[1]['CODICE SITO'])
                except:
                    index = [i for i, _ in enumerate(areas) if
                             (_['properties']['DENOMINAZI']).replace(" ", "") == (line[1]['SITO']).replace(" ", "")][0]
                    code = areas[index]["properties"]["CODICE"]

                id = str(code) + "_" + date

                if not any(d['id'] == id for d in measures):
                    measures.append({"id": id, "year": year, "datetime": date, "code": code, "outcome": [outcome]})
                else:
                    index = [i for i, _ in enumerate(measures) if _["id"] == id][0]
                    measures[index]["outcome"].append(outcome)

    for measure in measures:
        try:
            if cfg.get('variables', 'TYPE') == "MEAN":
                measure["outcome"] = round(statistics.mean(measure["outcome"]), 2)
            elif cfg.get('variables', 'TYPE') == "MAX":
                measure["outcome"] = max(measure["outcome"])
            elif cfg.get('variables', 'TYPE') == "MEDIAN":
                measure["outcome"] = round(statistics.median(measure["outcome"]), 2)
            else:
                measure["outcome"] = max(measure["outcome"])
        except:
            measure["outcome"] = measure["outcome"][0]

    return measures


def get_index_lat_long(lat, long, delta_lat, delta_long, min_lat, max_lat, min_long, max_long):
    index_min_lat = round((min_lat - lat[0]) / delta_lat)
    index_max_lat = round((max_lat - lat[0]) / delta_lat)
    index_min_long = round((min_long - long[0]) / delta_long)
    index_max_long = round((max_long - long[0]) / delta_long)

    return index_min_lat, index_max_lat, index_min_long, index_max_long


# MAX CONCENTRATION IN AN AREA IN A SPECIFIC HOUR
def getConc(url, lat, long, index_min_lat, index_min_long, area_poly):
    try:
        dataset = netCDF4.Dataset(url)

        concentration = dataset['conc'][0][0]

        values = []

        for i in range(0, len(concentration)):
            for j in range(0, len(concentration[0])):
                current_value = concentration[i][j]
                if cfg.getboolean('variables', 'BUFFER'):
                    values.append(current_value)
                else:
                    point = Point(long[index_min_long + j], lat[index_min_lat + i])
                    if point.within(area_poly):
                        values.append(current_value)
        if cfg.get('variables', 'TYPE') == "MEAN":
            value = round(statistics.mean(values), 2)
        elif cfg.get('variables', 'TYPE') == "MAX":
            value = max(values)
        elif cfg.get('variables', 'TYPE') == "MEDIAN":
            value = round(statistics.median(values), 2)
        else:
            value = "NaN"

    except Exception as e:
        print(e)
        value = "NaN"

    return value


# WORKER
def worker(areas, lat, long, delta_lat, delta_long, measure):
    index = [i for i, _ in enumerate(areas) if
             str(_['properties']['CODICE']) == str(measure['code'])][0]

    bbox = areas[index]['bbox']
    min_long = bbox[0]
    min_lat = bbox[1]
    max_long = bbox[2]
    max_lat = bbox[3]

    coordinates = areas[index]['geometry']['coordinates'][0][0]
    area_poly = Polygon(coordinates)

    if cfg.getboolean('variables', 'BUFFER'):
        max_lat, max_long = addKM2Coordinates(max_lat, max_long, 45)
        min_lat, min_long = addKM2Coordinates(min_lat, min_long, 225)

    id = measure["id"]

    date = measure['datetime']

    sample = measure['outcome']

    year = measure['year']

    _dataset = {"features": [], "label": sample, "id": id, 'year': year}
    features = []

    print(measure)

    for i in range(cfg.getint('variables', 'START'), cfg.getint('variables', 'END') + 1):
        reference_hour = datetime.strptime(date, '%d/%m/%y/%H:%M') - timedelta(hours=i)
        formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

        year = str(reference_hour.year)
        month = '{:>02d}'.format(reference_hour.month)
        day = '{:>02d}'.format(reference_hour.day)

        index_min_lat, index_max_lat, index_min_long, index_max_long = get_index_lat_long(lat, long,
                                                                                          delta_lat, delta_long,
                                                                                          min_lat, max_lat,
                                                                                          min_long, max_long)

        url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + \
              ".nc?conc[0:1:0][0:1:1][" + str(index_min_lat) + ":1:" + str(index_max_lat) + "][" + \
              str(index_min_long) + ":1:" + str(index_max_long) + "]"

        value = getConc(url, lat, long, index_min_lat, index_min_long, area_poly)
        # value = np.random.randint(150)

        print(id, value, reference_hour)

        features.append(str(value))

    _dataset["features"] = features

    print(_dataset)

    return _dataset


# CREATE DATASET
def create_dataset(areas, lat, long, delta_lat, delta_long, max_measures):
    my_dataset = []

    workers = multiprocessing.cpu_count()
    func = partial(worker, areas, lat, long, delta_lat, delta_long)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(func, max_measures):
            my_dataset.append(result)

    return my_dataset


# ADD KM TO COORDINATES
def addKM2Coordinates(latitude, longitude, bearing):
    R = 6378.1
    d = cfg.getfloat('variables', 'RANGE_BUFFER')
    bearing = math.radians(bearing)

    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lat2, lon2


# GET LAT/LONG ARRAY ONLY ONE TIME
def get_lat_long():
    url = "http://193.205.230.6:8080/opendap/opendap/wcm3/d03/archive/2020/01/01/wcm3_d03_20200101Z0000.nc"
    _dataset = netCDF4.Dataset(url)
    long = _dataset.variables['longitude']
    lat = _dataset.variables['latitude']

    N = len(lat)
    sum_delta_lat = 0
    for i in range(1, N):
        sum_delta_lat += lat[i] - lat[i - 1]
    delta_lat = sum_delta_lat / N

    M = len(long)
    sum_delta_long = 0
    for i in range(1, M):
        sum_delta_long += long[i] - long[i - 1]
    delta_long = sum_delta_long / M

    return np.array(lat), np.array(long), delta_lat, delta_long


if __name__ == "__main__":
    start = time.time()

    areas = load_areas()
    measures = remove_measures_duplicates(areas)

    lat, long, delta_lat, delta_long = get_lat_long()
    dataset = create_dataset(areas, lat, long, delta_lat, delta_long, measures)

    list = cfg.get('variables', 'YEAR')
    years = json.loads(list)
    with open('dataset_' + cfg.get('variables', 'TYPE').lower() + '_'.join(map(str, years)) + '.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f)

    end = time.time()

    print(end-start)