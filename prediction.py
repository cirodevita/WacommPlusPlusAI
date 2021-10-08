from netCDF4 import Dataset
import sys
from datetime import datetime, timedelta
import pickle
import numpy as np
from shapely.geometry import Polygon, Point
from tensorflow import keras
from main import load_areas, get_lat_long, worker, get_index_lat_long
from training import KnnDtw
import matplotlib.pyplot as plt
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('config.ini')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _datetime = sys.argv[1]

        print("Loading areas...")
        areas = load_areas()
        print("Loading completed!")

        print("Loading lat/long, conc...")
        lat, long, delta_lat, delta_long = get_lat_long()

        src = cfg.get('variables', 'URL') + _datetime[0:4] + "/" + _datetime[4:6] + "/" + _datetime[6:8] + \
              "/wcm3_d03_" + _datetime + ".nc"
        dst = "test/wcm3_d03_" + _datetime + ".nc.nc4"

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

        new_datetime = _datetime[6:8] + "/" + _datetime[4:6] + "/" + _datetime[2:4] + "/" + _datetime[9:11] + ":00"
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
            _dataset = Dataset(url)
            concentration = _dataset['conc'][0][0]

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
        print(dataset)
        print("Dataset Created!")

        print("Prediction")

        for pixel in dataset:
            features = pixel["features"]
            model = pickle.load(open('KNeighborsClassifier', 'rb'))
            y_pred_knn = model.predict(np.array([features, ]))
            print("[KNeighborsClassifier] Class: ", int(y_pred_knn[0]))
            print("End Prediction!")

            i = pixel["id"].split("_")[0]
            j = pixel["id"].split("_")[1]

            classVar[0, i, j] = y_pred_knn + 1

        print("End Prediction!")

        ncdstfile.close()
        print("NetCDF created!")

        """
        for area in areas:
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
            print(index_min_lat, index_max_lat, index_min_long, index_max_long)

            timeVar[:] = ncsrcfile.variables["time"][:]
            depthVar[:] = [0, -2.5, -5, -10, -15, -20, -25, -50, -100, -200, -300]
            lonVar[:] = long
            latVar[:] = lat
            concVar[:] = ncsrcfile.variables["conc"][:]
            sfconcVar[:] = ncsrcfile.variables["sfconc"][:]

            url = cfg.get('variables', 'URL') + _datetime[0:4] + "/" + _datetime[4:6] + "/" + _datetime[6:8] + \
                  "/wcm3_d03_" + _datetime + ".nc?conc[0:1:0][0:1:1][" + \
                  str(index_min_lat) + ":1:" + str(index_max_lat) + "][" + \
                  str(index_min_long) + ":1:" + str(index_max_long) + "]"
            dataset = Dataset(url)

            concentration = dataset['conc'][0][0]

            for i in range(index_min_lat, index_max_lat):
                for j in range(index_min_long, index_max_long):
                    point = Point(long[j], lat[i])
                    if point.within(area_poly):
                        features = []
                        for k in range(0, 168):
                            value = concentration[i-index_max_lat][j-index_max_long]
                            features.append(value)

                        print("Prediction...")
                        model = pickle.load(open('KNeighborsClassifier', 'rb'))
                        y_pred_knn = model.predict(np.array([features, ]))
                        print("[KNeighborsClassifier] Class: ", int(y_pred_knn[0]))
                        print("End Prediction!")

                        classVar[0, i, j] = y_pred_knn + 1

        print("Dataset created!")
        ncdstfile.close()
        print("NetCDF created!")
        """

        # PER BANK
        """
            code = area['properties']['CODICE']
            measure = {"id": str(code) + "_" + new_datetime[:-6], "year": _datetime[0:4], "datetime": new_datetime,
                       "code": code, "outcome": "--"}
            dataset = worker(areas, lat, long, delta_lat, delta_long, measure)
            features = list(np.float_(dataset["features"]))

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
            plt.title(str(code))
            plt.show()

            print("Prediction...")
            model = keras.models.load_model('CNN')
            y_pred_cnn = np.argmax(model.predict([np.array([features, ]), np.array([features, ]), np.array([features, ])]),
                                   axis=1)

            model = pickle.load(open('KNeighborsClassifier', 'rb'))
            y_pred_knn = model.predict(np.array([features, ]))

            model = pickle.load(open('KNN_DTW', 'rb'))
            y_pred_knn_dtw, _ = model.predict(np.array([features, ]))

            print("[CNN] Class: ", y_pred_cnn[0])
            print("[KNeighborsClassifier] Class: ", int(y_pred_knn[0]))
            print("[KNN_DTW] Class: ", int(y_pred_knn_dtw[0]))

            print("End Prediction!")

            
            print("Creating NetCDF...")

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

            timeVar[:] = ncsrcfile.variables["time"][:]
            depthVar[:] = [0, -2.5, -5, -10, -15, -20, -25, -50, -100, -200, -300]
            lonVar[:] = long
            latVar[:] = lat
            concVar[:] = conc
            sfconcVar[:] = sfconc

            for i in range(index_min_lat, index_max_lat):
                for j in range(index_min_long, index_max_long):
                    point = Point(long[j], lat[i])
                    if point.within(area_poly):
                        classVar[0, i, j] = y_pred_knn + 1

        ncdstfile.close()
        print("NetCDF created!")
        """

    else:
        print("Usage: datetime")
