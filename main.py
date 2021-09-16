import pandas as pd
import json
import netCDF4
from datetime import datetime, timedelta
from configparser import ConfigParser
from shapely.geometry import Point, Polygon
from multiprocessing.pool import ThreadPool as Pool


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
        if line[1]['ANNO ACCETTAZIONE'] == cfg.getint('variables', 'YEAR'):
            id = line[1]['NUMERO SCHEDA']
            date = line[1]['DATA PRELIEVO'] + cfg.get('variables', 'SAMPLE_HOUR')
            outcome = line[1]['ESITO']
            site = line[1]['SITO']
            lat = line[1]['LATITUDINE_DEF']
            lon = line[1]['LONGITUDINE_DEF']
            year = line[1]['ANNO ACCETTAZIONE']

            if not any(d['id'] == id for d in max_measures):
                max_measures.append({"id": id, 'site_name': site, 'datetime': date, 'latitude': lat, 'longitude': lon,
                                     'outcome': outcome, 'year': year})
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

    return index_min_lat, index_max_lat, index_min_long, index_max_long


def getMaxConc(f, url, min_lat, max_lat, min_long, max_long, area_poly):
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
            for i in range(index_min_lat, index_max_lat + 1):
                for j in range(index_min_long, index_max_long + 1):
                    point = Point(long[j], lat[i])
                    if point.within(area_poly):
                        value = concentration[k][i][j]
                        if value > max:
                            max = value
    except:
        f.write(url + '\n')
        f.flush()
        max = "NaN"

    return max


# WORKER
def worker(dataset, measure, file):
    index = [i for i, _ in enumerate(areas) if (_['properties']['DENOMINAZI']).replace(" ", "") == (measure['site_name']).replace(" ", "")][0]

    bbox = areas[index]['bbox']
    min_long = bbox[0]
    min_lat = bbox[1]
    max_long = bbox[2]
    max_lat = bbox[3]

    coordinates = areas[index]['geometry']['coordinates'][0][0]

    area_poly = Polygon(coordinates)

    date = measure['datetime']

    sample = measure['outcome']

    year = measure['year']

    _dataset = {"features": [], "label": sample, "site": measure['site_name'], 'year': year}
    features = []

    print(measure)

    for i in range(cfg.getint('variables', 'START'), cfg.getint('variables', 'END') + 1):
        reference_hour = datetime.strptime(date, '%d/%m/%y/%H:%M') - timedelta(hours=i)
        formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

        year = str(reference_hour.year)
        month = '{:>02d}'.format(reference_hour.month)
        day = '{:>02d}'.format(reference_hour.day)

        url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + ".nc"

        max = getMaxConc(file, url, min_lat, max_lat, min_long, max_long, area_poly)

        features.append(str(max))

    _dataset["features"] = features

    my_dataset.append(_dataset)


# CREATE DATASET
def create_dataset(areas, max_measures):
    my_dataset = []
    f = open("test/url_mancanti.txt", "a")

    pool_size = 16
    pool = Pool(pool_size)  

    for measure in max_measures:
        pool.apply_async(worker, (my_dataset, measure, f,))

    pool.close()
    pool.join()
    f.close()

    return my_dataset


if __name__ == "__main__":
    areas = load_areas()
    max_measures = remove_measures_duplicates()
    dataset = create_dataset(areas, max_measures)

    with open('dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f)
