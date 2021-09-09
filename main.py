import pandas as pd
import json
import netCDF4
from datetime import datetime, timedelta
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('config.ini')

max_measures = []

# LOAD AREAS INFO
with open("files/banchi.geojson") as input_file:
    data = json.load(input_file)
    areas = data["features"]

# REMOVE MEASURES DUPLICATES
lines = pd.read_csv('files/misurazioni.csv', delimiter=';')
for line in lines.iterrows():
    id = line[1]['NUMERO SCHEDA']
    date = line[1]['DATA PRELIEVO'] + "/10:00"
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

# CREATE DATASET
for measure in max_measures:
    index = [i for i, _ in enumerate(areas) if _['properties']['DENOMINAZI'] == measure['site_name']][0]
    bbox = areas[index]['bbox']
    date = measure['datetime']

    for i in range(int(cfg.get('variables', 'START')), int(cfg.get('variables', 'END')) + 1):
        reference_hour = datetime.strptime(date, '%d/%m/%y/%H:%M') - timedelta(hours=i)
        formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

        year = str(reference_hour.year)
        month = f'{reference_hour.month:02}'
        day = f'{reference_hour.day:02}'

        url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + ".nc"
        print(url)
        '''
        try:
            dataset = netCDF4.Dataset(url)
            # conc = dataset.variables['conc']
        except:
            print(url)
        '''