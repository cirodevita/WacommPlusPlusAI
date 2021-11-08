import pandas as pd
from netCDF4 import Dataset
import sys
import os
from datetime import datetime, timedelta
import pickle
import numpy as np
from shapely.geometry import Polygon, Point
from tensorflow import keras
from main import load_areas, get_lat_long, worker, get_index_lat_long
# from training import KnnDtw
# import matplotlib.pyplot as plt
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('config.ini')


def get_datatime(file):
    arr = []

    lines = pd.read_csv(file, delimiter=';')
    for line in lines.iterrows():
        date = str(line[1]['DATA PRELIEVO']) + cfg.get('variables', 'SAMPLE_HOUR')
        if not any(d == date for d in arr):
            arr.append(date)

    return arr


def prediction(lat, long, delta_lat, delta_long, _datetime):
    src = cfg.get('variables', 'URL') + _datetime[0:4] + "/" + _datetime[4:6] + "/" + _datetime[6:8] + \
          "/wcm3_d03_" + _datetime + ".nc"
    dst = "output/wcm3_d03_" + _datetime + ".nc.nc4"

    ncsrcfile = Dataset(src)

    ncols = len(long)
    nrows = len(lat)

    print("Loading completed!")

    ncdstfile = Dataset(dst, "w", format="NETCDF4")
    timeDim = ncdstfile.createDimension("time", size=1)
    depthDim = ncdstfile.createDimension("depth", size=11)
    latDim = ncdstfile.createDimension("latitude", size=nrows)
    lonDim = ncdstfile.createDimension("longitude", size=ncols)

    timeVar = ncdstfile.createVariable("time", "i4", "time")
    timeVar.description = "Time since initialization"
    timeVar.long_name = "time since initialization"
    timeVar.units = "seconds since 1968-05-23 00:00:00"
    timeVar.calendar = "gregorian"
    timeVar.field = "time, scalar, series"

    depthVar = ncdstfile.createVariable("depth", "f4", "depth")
    depthVar.description = "depth"
    depthVar.long_name = "depth"
    depthVar.units = "meters"

    lonVar = ncdstfile.createVariable("longitude", "f4", "longitude")
    lonVar.description = "Longitude"
    lonVar.long_name = "longitude"
    lonVar.units = "degrees_east"

    latVar = ncdstfile.createVariable("latitude", "f4", "latitude")
    latVar.description = "Latitude"
    lonVar.long_name = "latitude"
    latVar.units = "degrees_north"

    concVar = ncdstfile.createVariable("conc", "f4", ("time", "depth", "latitude", "longitude"), fill_value=1.e+37)
    concVar.description = "concentration of suspended matter in sea water"
    concVar.units = "1"
    concVar.long_name = "concentration"

    sfconcVar = ncdstfile.createVariable("sfconc", "f4", ("time", "latitude", "longitude"), fill_value=1.e+37)
    sfconcVar.description = "concentration of suspended matter at the surface"
    sfconcVar.units = "1"
    sfconcVar.long_name = "surface_concentration"

    classVar = ncdstfile.createVariable("class_predict", "i1", ("time", "latitude", "longitude"), fill_value=0)
    classVar.description = "predicted class of concentration of pollutants in mussels"
    classVar.long_name = "class_predict"

    timeVar[:] = ncsrcfile.variables["time"][:]
    depthVar[:] = [0, -2.5, -5, -10, -15, -20, -25, -50, -100, -200, -300]
    lonVar[:] = long
    latVar[:] = lat
    concVar[:] = ncsrcfile.variables["conc"][:]
    sfconcVar[:] = ncsrcfile.variables["sfconc"][:]

    print("Creating Dataset...")

    dataset = []

    for h in range(cfg.getint('variables', 'START'), cfg.getint('variables', 'END') + 1):
        reference_hour = datetime.strptime(_datetime, "%Y%m%dZ%H%M") - timedelta(hours=h)
        formatted_hour = reference_hour.strftime("%Y%m%dZ%H%M")

        print(reference_hour)
        year = str(reference_hour.year)
        month = '{:>02d}'.format(reference_hour.month)
        day = '{:>02d}'.format(reference_hour.day)

        url = cfg.get('variables', 'URL') + year + "/" + month + "/" + day + "/wcm3_d03_" + formatted_hour + ".nc"
        try:
            _dataset = Dataset(url)
            concentration = _dataset['conc'][0][0]
        except:
            concentration = np.zeros((nrows, ncols))

        for area in areas:
            code = area['properties']['CODICE']

            bbox = area['bbox']
            min_long = bbox[0]
            min_lat = bbox[1]
            max_long = bbox[2]
            max_lat = bbox[3]

            coordinates = area['geometry']['coordinates'][0][0]

            area_poly = Polygon(coordinates)

            index_min_lat, index_max_lat, index_min_long, index_max_long = get_index_lat_long(lat, long,
                                                                                              delta_lat, delta_long,
                                                                                              min_lat, max_lat,
                                                                                              min_long, max_long)
            for i in range(index_min_lat, index_max_lat):
                for j in range(index_min_long, index_max_long):
                    point = Point(long[j], lat[i])
                    if point.within(area_poly):
                        id = str(i) + "_" + str(j) + "_" + str(code)
                        if not any(d['id'] == id for d in dataset):
                            dataset.append({"id": id, "features": [concentration[i][j]]})
                        else:
                            index = [i for i, _ in enumerate(dataset) if _["id"] == id][0]
                            dataset[index]["features"].append(concentration[i][j])
    # print(dataset)
    print("Dataset Created!")

    print("Prediction...")

    for pixel in dataset:
        features = pixel["features"]
        model = pickle.load(open('KNeighborsClassifier', 'rb'))
        y_pred_knn = model.predict(np.array([features, ]))
        # print("[KNeighborsClassifier] Class: ", int(y_pred_knn[0]))
        print("End Prediction!")

        i = pixel["id"].split("_")[0]
        j = pixel["id"].split("_")[1]

        classVar[0, i, j] = y_pred_knn + 1

    print("End Prediction!")

    ncdstfile.close()
    print("NetCDF created!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Loading areas...")
        areas = load_areas()
        print("Loading completed!")

        print("Loading lat/long, conc...")
        lat, long, delta_lat, delta_long = get_lat_long()

        if os.path.exists(sys.argv[1]):
            array = get_datatime(sys.argv[1])
            for date in array:
                try:
                    formatted_hour = datetime.strptime(date, '%d/%m/%y/%H:%M').strftime("%Y%m%dZ%H%M")
                    prediction(lat, long, delta_lat, delta_long, formatted_hour)
                except:
                    print("Error!")
                    continue
        else:
            try:
                _datetime = sys.argv[1]
                prediction(lat, long, delta_lat, delta_long, _datetime)
            except:
                print("Error!")
                pass

        """
        # new_datetime = _datetime[6:8] + "/" + _datetime[4:6] + "/" + _datetime[2:4] + "/" + _datetime[9:11] + ":00"
        # measure = {"id": "1500041" + "_" + new_datetime[:-6], "year": _datetime[0:4], "datetime": new_datetime,
        #           "code": "1500041", "outcome": "--"}
        # dataset = worker(areas, lat, long, delta_lat, delta_long, measure)
        # print("Dataset created!")
        # features = list(np.float_(dataset["features"]))

        features = [0.0, 0.0, 0.0, 709.5, 712.0, 102.5, 1138.0, 1031.5, 0.0, 1254.5, 717.0, 0.0, 94.0, 73.0, 490.0,
                    439.0, 287.0, 275.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 750.5, 770.0, 449.5, 793.0, 745.5,
                    0.0, 78.5, 88.0, 0.0, 0.0, 0.0, 536.5, 557.0, 531.5, 336.5, 0.0, 119.5, 602.5, 636.0, 0.0, 669.0,
                    473.0, 0.0, 0.0, 188.5, 279.0, 0.0, 943.5, 967.5, 1.0, 1003.5, 887.5, 0.0, 100.5, 0.0, 370.0, 531.0,
                    777.5, 185.5, 0.0, 0.0, 260.0, 10.5, 0.0, 0.0, 0.0, 0.0, 0.0, 709.0, 367.0, 0.0, 1139.0, 997.0, 0.0,
                    1030.5, 934.0, 0.0, 0.0, 0.0, 339.5, 319.5, 273.5, 97.0, 0.0, 0.0, 347.5, 255.5, 0.0, 2.5, 0.0, 0.0,
                    0.0, 603.0, 397.5, 0.0, 1104.0, 1166.0, 0.0, 1348.0, 1230.0, 675.0, 2.5, 0.0, 870.0, 992.0, 1122.0,
                    1429.0, 0.0, 104.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 637.0, 567.0, 0.0, 1285.0, 1282.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 3.5, 1404.5, 1395.0, 1481.0, 1592.5, 275.5, 583.5, 280.0, 2.5, 0.0, 0.0, 0.0, 0.0,
                    0.0, 215.0, 256.5, 0.0, 1878.5, 445.5, 0.0, 2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 1.5]
        datetime_arr = []
        for i in range(0, len(features)):
            reference_hour = datetime(int(_datetime[0:4]), int(_datetime[4:6]), int(_datetime[6:8]),
                                      int(_datetime[9:11]), 0) - timedelta(hours=i)
            datetime_arr.append(reference_hour)

        plt.plot(datetime_arr, features)
        plt.ylim(0, 520)
        plt.axhline(y=67, color='b', linestyle='-', linewidth=0.2)
        plt.axhline(y=230, color='b', linestyle='-', linewidth=0.2)
        plt.xticks([datetime_arr[0], datetime_arr[-1]])
        plt.show()

        print("Prediction...")

        model_cnn = keras.models.load_model('CNN')
        y_pred_cnn = np.argmax(model_cnn.predict([np.array([features, ]), np.array([features, ]), np.array([features, ])]), axis=1)

        model_resnet = keras.models.load_model('ResNet')
        y_pred_resnet = np.argmax(model_resnet.predict(np.array([features, ])), axis=1)

        model_2knn = pickle.load(open('KNeighborsClassifier', 'rb'))
        y_pred_knn = model_2knn.predict(np.array([features, ]))

        model_knn_dtw = pickle.load(open('KNN_DTW', 'rb'))
        y_pred_knn_dtw, _ = model_knn_dtw.predict(np.array([features, ]))

        print("[CNN] Class: ", y_pred_cnn[0])
        print("[ResNet] Class: ", y_pred_resnet[0])

        print("[KNeighborsClassifier] Class: ", int(y_pred_knn[0]))
        print("[KNN_DTW] Class: ", int(y_pred_knn_dtw[0]))
        """
    else:
        print("Usage: datetime")
