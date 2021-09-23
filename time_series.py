import concurrent.futures
import multiprocessing
import time
import calendar
from configparser import ConfigParser
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import netCDF4
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon

from main import load_areas, remove_measures_duplicates, get_lat_long, get_index_lat_long, getMaxConc

cfg = ConfigParser()
cfg.read('config.ini')


def measure(index, date, reference_hour, area, max_measures):
    if any((d['datetime'] == date) and (
            (d['site_name'].replace(" ", "")) == (area['properties']['DENOMINAZI'].replace(" ", ""))) for d in
           max_measures):
        _index = [i for i, _ in enumerate(max_measures) if
                  ((_['datetime']) == date) and ((_['site_name'].replace(" ", "")) == (
                      area['properties']['DENOMINAZI'].replace(" ", "")))][0]

        time_series[index]['datetime_measures'].append(reference_hour)
        time_series[index]['measures'].append(int(max_measures[_index]['outcome']))


def worker(year, areas, lat, long, delta_lat, delta_long, max_measures, time_series, i):
    reference_hour = datetime(year, 1, 1, 0, 0) + timedelta(hours=i)
    formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

    _year = str(reference_hour.year)
    _month = f'{reference_hour.month:02}'
    _day = f'{reference_hour.day:02}'
    _hour = f'{reference_hour.hour:02}'

    date = _day + "/" + _month + "/" + _year[2:4] + "/" + _hour + ":00"

    for area in areas:
        bbox = area['bbox']
        min_long = bbox[0]
        min_lat = bbox[1]
        max_long = bbox[2]
        max_lat = bbox[3]

        coordinates = area['geometry']['coordinates'][0][0]
        area_poly = Polygon(coordinates)

        index = [i for i, _ in enumerate(time_series) if
                 (_['name']).replace(" ", "") == (area['properties']['DENOMINAZI']).replace(" ", "")][0]

        time_series[index]['datetime'].append(reference_hour)

        index_min_lat, index_max_lat, index_min_long, index_max_long = get_index_lat_long(lat, long,
                                                                                          delta_lat, delta_long,
                                                                                          min_lat, max_lat,
                                                                                          min_long, max_long)

        url = cfg.get('variables', 'URL') + _year + "/" + _month + "/" + _day + "/wcm3_d03_" + formatted_hour + \
              ".nc?conc[0:1:0][0:1:1][" + str(index_min_lat) + ":1:" + str(index_max_lat) + "][" + \
              str(index_min_long) + ":1:" + str(index_max_long) + "]"

        # max = getMaxConc(url, lat, long, index_min_lat, index_min_long, area_poly)
        max = np.random.randint(300)

        if max == "NaN":
            max = 0
        time_series[index]['values'].append(int(max))

        measure(index, date, reference_hour, area, max_measures)


def create_timeseries(year, areas, lat, long, delta_lat, delta_long, max_measures, time_series, hours):
    """
    workers = multiprocessing.cpu_count()
    func = partial(worker, year, areas, lat, long, delta_lat, delta_long, max_measures, time_series)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        [executor.submit(func, i) for i in range(0, hours)]
    """
    for i in range(0, hours + 1):
        worker(year, areas, lat, long, delta_lat, delta_long, max_measures, time_series, i)


def create_graphics(time_series):
    for time in time_series:
        directory_name = "graphics/" + (time["name"]).replace("/", "")
        Path(directory_name).mkdir(parents=True, exist_ok=True)

        hours = []
        measures = []

        temp_month = 1
        temp_count = 0
        index_start = 0

        for t in time['datetime']:
            if temp_month == int(t.month):
                year = t.year
                hours.append(t)
            else:
                x = np.array(hours)
                index_end = index_start + len(x)
                y = np.array(time['values'][index_start:index_end])

                count = 0
                for m in time['datetime_measures']:
                    if temp_month == m.month:
                        measures.append(m)
                        count += 1

                x1 = np.array(measures)
                y1 = np.array(time['measures'][temp_count:temp_count + count])

                temp_count += count

                fig = plt.figure()
                fig.set_size_inches(10, 5.0)
                fig.suptitle(time['name'] + ": " + calendar.month_name[int(temp_month)] + " " + str(year), fontsize=20)

                ax = fig.add_subplot()
                ax.plot(x, y, linewidth=0.2, color='black')
                ax.axhline(y=67, color='b', linestyle='-', linewidth=0.2)
                ax.axhline(y=230, color='b', linestyle='-', linewidth=0.2)
                ax.set_ylim(0, 520)
                ax.set_xlabel("Time [h]")
                ax.set_ylabel("number of particles")
                ax.set_xticks([x[0], x[-1]])
                ax.set_yticks([67, 230, 500])

                ax2 = ax.twinx()
                ax2.bar(x1, y1, width=0.2, color='red')
                ax2.set_ylim(0, 520)
                ax2.set_ylabel("E. coil/100g FIL")
                ax.set_xticks([x[0], x[-1]])
                ax2.set_yticks([67, 230, 500])

                if len(x1) != 0:
                    for k in x1:
                        plt.fill_between(x, 0, y, color='green',
                                         where=(x < k) & (x > k - timedelta(hours=168)))

                fig.savefig(directory_name + "/" + str(calendar.month_name[int(temp_month)]) + "-" + str(year) + ".png",
                            dpi=100)

                hours.clear()
                measures.clear()

                temp_month = int(t.month)
                index_start += len(x) + 1


if __name__ == "__main__":
    start = time.time()

    areas = load_areas()
    max_measures = remove_measures_duplicates()

    time_series = []
    for area in areas:
        time_series.append({"name": area['properties']['DENOMINAZI'], "datetime": [], "values": [],
                            "datetime_measures": [], "measures": []})

    lat, long, delta_lat, delta_long = get_lat_long()

    year = cfg.getint('time_series', 'YEAR')
    dt = datetime(year, 1, 1, 00, 00, 00)
    # dt2 = datetime(year + 1, 1, 1, 00, 00, 00)
    dt2 = datetime(year, 2, 1, 00, 00, 00)
    hours = (dt2 - dt).days * 24
    create_timeseries(year, areas, lat, long, delta_lat, delta_long, max_measures, time_series, hours)

    create_graphics(time_series)

    end = time.time()

    print(end - start)
