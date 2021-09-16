import matplotlib.pyplot as plt
import datetime
import numpy as np
from configparser import ConfigParser
from shapely.geometry import Polygon
import calendar

from main import load_areas, getMaxConc, remove_measures_duplicates

cfg = ConfigParser()
cfg.read('config.ini')

areas = load_areas()
max_measures = remove_measures_duplicates()

_year = 2019
_month = 1
_day = 1

_hours = 8760

for area in areas:
    bbox = area['bbox']

    coordinates = area['geometry']['coordinates'][0][0]
    area_poly = Polygon(coordinates)

    hours = []
    array = []

    datetime_measures = []
    measures = []

    temp_month = 1

    for i in range(0, _hours + 1):
        reference_hour = datetime.datetime(_year, _month, _day, 0, 0) + datetime.timedelta(hours=i)
        formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

        hours.append(reference_hour)

        year = str(reference_hour.year)
        month = f'{reference_hour.month:02}'
        day = f'{reference_hour.day:02}'
        hour = f'{reference_hour.hour:02}'

        if temp_month == int(month):
            url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + ".nc"

            date = day + "/" + month + "/" + year[2:4] + "/" + hour + ":00"

            if any((d['datetime'] == date) and ((d['site_name'].replace(" ", "")) == (area['properties']['DENOMINAZI'].replace(" ", ""))) for d in max_measures):
                datetime_measures.append(reference_hour)
                measures.append(np.random.randint(300))

            # max = getMaxConc(url, bbox[1], bbox[3], bbox[0], bbox[2], area_poly)

            # array.append(max)
        else:
            x = np.array(hours)
            y = np.random.randint(300, size=x.shape)
            # y = np.array(array)

            x1 = np.array(datetime_measures)
            y1 = np.array(measures)

            fig = plt.figure()
            fig.suptitle(area['properties']['DENOMINAZI'] + " " + calendar.month_name[int(temp_month)] + " " + year,
                         fontsize=20)

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
                plt.fill_between(x, 0, y, color='green', where=(x < x1[-1]) & (x > x1[-1] - datetime.timedelta(hours=168)))

            plt.show()

            hours = []
            array = []

            datetime_measures = []
            measures = []

            temp_month = int(month)
