import matplotlib.pyplot as plt
import datetime
import numpy as np
from configparser import ConfigParser
from shapely.geometry import Polygon
import calendar
from pathlib import Path

from main import load_areas, getMaxConc, remove_measures_duplicates

cfg = ConfigParser()
cfg.read('config.ini')

areas = load_areas()
max_measures = remove_measures_duplicates()


time_series = []
for area in areas:
    time_series.append({"name": area['properties']['DENOMINAZI'], "datetime": [], "values": [],
                        "datetime_measures": [], "measures": []})

_year = 2019
_month = 1
_day = 1

_hours = 8760

file = open("test/url_mancanti.txt", "a")

for i in range(0, _hours + 1):
    reference_hour = datetime.datetime(_year, _month, _day, 0, 0) + datetime.timedelta(hours=i)
    formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

    year = str(reference_hour.year)
    month = f'{reference_hour.month:02}'
    day = f'{reference_hour.day:02}'
    hour = f'{reference_hour.hour:02}'

    url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + ".nc"

    date = day + "/" + month + "/" + year[2:4] + "/" + hour + ":00"

    for area in areas:
        bbox = area['bbox']

        coordinates = area['geometry']['coordinates'][0][0]
        area_poly = Polygon(coordinates)

        index = [i for i, _ in enumerate(time_series) if
                 (_['name']).replace(" ", "") == (area['properties']['DENOMINAZI']).replace(" ", "")][0]

        time_series[index]['datetime'].append(reference_hour)
        max = getMaxConc(file, url, bbox[1], bbox[3], bbox[0], bbox[2], area_poly)
        # max = np.random.randint(300)
        if max == "NaN":
            max = 0
        time_series[index]['values'].append(int(max))

        if any((d['datetime'] == date) and (
                (d['site_name'].replace(" ", "")) == (area['properties']['DENOMINAZI'].replace(" ", ""))) for d in
               max_measures):
            _index = [i for i, _ in enumerate(max_measures) if
                      ((_['datetime']) == date) and ((_['site_name'].replace(" ", "")) == (
                          area['properties']['DENOMINAZI'].replace(" ", "")))][0]

            time_series[index]['datetime_measures'].append(reference_hour)
            time_series[index]['measures'].append(int(max_measures[_index]['outcome']))

file.close()

for time in time_series:
    directory_name = "graphics/" + (time["name"]).replace("/", "")
    Path(directory_name).mkdir(parents=True, exist_ok=True)

    hours = []
    measures = []

    temp_month = 1
    temp_count = 0

    for t in time['datetime']:
        if temp_month == int(t.month):
            year = t.year
            hours.append(t)
        else:
            x = np.array(hours)
            y = np.array(time['values'][(t.month - 1) * len(x):t.month * len(x)])

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
                    plt.fill_between(x, 0, y, color='green', where=(x < k) & (x > k - datetime.timedelta(hours=168)))

            fig.savefig(directory_name + "/" + str(calendar.month_name[int(temp_month)]) + "-" + str(year) + ".png",
                        dpi=100)

            hours.clear()
            measures.clear()

            temp_month = int(t.month)