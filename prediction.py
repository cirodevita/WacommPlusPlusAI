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
    if len(sys.argv) > 2:
        _datetime = sys.argv[1]
        code = sys.argv[2]

        print("Loading areas...")
        areas = load_areas()
        print("Loading completed!")

        print("Loading lat/long...")
        lat, long, delta_lat, delta_long = get_lat_long()
        print("Loading completed!")

        print("Dataset creation...")
        new_datetime = _datetime[6:8] + "/" + _datetime[4:6] + "/" + _datetime[2:4] + "/" + _datetime[9:11] + ":00"
        measure = {"id": str(code) + "_" + new_datetime[:-6], "year": _datetime[0:4], "datetime": new_datetime,
                   "code": code, "outcome": "--"}
        dataset = worker(areas, lat, long, delta_lat, delta_long, measure)
        print("Dataset created!")
        features = list(np.float_(dataset["features"]))

        # features = [0.0, 0.0, 0.0, 15.0, 15.0, 10.0, 0.0, 130.0, 156.5, 341.0, 311.0, 299.0, 312.0, 298.5, 257.0, 170.0, 114.5, 61.0, 9.0, 0.5, 0.0, 2.0, 12.0, 30.0, 0.0, 153.0, 0.0, 138.5, 133.0, 143.5, 165.0, 150.0, 101.0, 127.0, 100.5, 107.5, 103.5, 95.5, 113.5, 75.0, 4.5, 4.5, 5.5, 0.0, 0.0, 164.0, 133.0, 126.0, 0.0, 233.0, 0.0, 388.0, 454.0, 601.5, 642.0, 695.0, 668.0, 354.5, 981.5, 1067.5, 1048.0, 919.0, 815.0, 652.0, 454.0, 432.0, 454.0, 366.5, 276.5, 468.5, 411.5, 351.0, 0.0, 98.5, 0.0, 316.0, 202.5, 137.5, 0.0, 140.5, 106.0, 66.5, 60.0, 65.5, 66.0, 57.5, 46.5, 37.0, 35.0, 39.5, 44.5, 40.0, 55.5, 111.5, 111.5, 102.0, 0.0, 11.0, 0.0, 88.5, 9.0, 12.0, 23.0, 78.5, 17.0, 14.0, 22.0, 14.5, 26.0, 42.5, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 83.0, 62.0, 42.5, 0.0, 54.0, 0.0, 123.0, 244.0, 341.5, 341.0, 346.0, 226.0, 164.0, 174.5, 203.0, 238.0, 230.0, 221.5, 206.5, 196.5, 156.5, 21.5, 0.0, 0.0, 316.5, 256.0, 91.5, 0.0, 83.0, 0.0, 375.5, 329.0, 375.0, 461.5, 541.0, 217.5, 182.5, 173.5, 186.5, 223.5, 215.0, 151.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 662.0, 679.0, 682.5]
        # features = [12.25, 12.69, 12.03, 11.36, 10.25, 8.84, 7.54, 6.27, 5.75, 5.86, 6.05, 5.91, 6.03, 6.19, 6.52, 7.09, 7.72, 8.87, 10.09, 11.65, 12.55, 13.08, 13.46, 16.14, 19.91, 21.03, 20.7, 19.41, 18.28, 17.65, 17.75, 17.93, 17.35, 16.93, 16.6, 16.25, 15.44, 14.59, 14.1, 13.41, 12.28, 11.16, 10.11, 9.27, 8.05, 7.27, 7.0, 8.01, 8.42, 8.18, 8.47, 7.81, 6.99, 6.31, 5.83, 5.73, 6.13, 5.61, 8.24, 7.49, 7.98, 6.46, 6.84, 5.72, 5.57, 4.18, 2.26, 6.23, 5.73, 2.03, 0.43, 5.94, 5.06, 1.67, 5.95, 4.6, 3.21, 0.06, 11.85, 10.82, 10.65, 5.1, 6.01, 5.38, 7.09, 3.82, 9.49, 10.88, 10.18, 7.64, 2.66, 20.51, 19.38, 7.14, 0.08, 9.05, 19.64, 1.01, 42.07, 37.95, 35.28, 7.62, 39.49, 29.24, 31.36, 37.51, 38.46, 36.23, 34.17, 25.85, 26.69, 22.73, 19.87, 12.52, 1.45, 18.42, 13.46, 2.54, 0.02, 8.65, 14.57, 0.51, 24.29, 19.93, 15.61, 2.31, 22.02, 15.95, 14.84, 13.08, 14.24, 14.45, 16.02, 18.1, 20.74, 22.91, 25.29, 25.99, 24.79, 22.22, 17.96, 11.73, 9.86, 10.29, 10.49, 10.58, 10.43, 10.68, 11.92, 12.94, 16.37, 24.35, 23.5, 19.64, 19.36, 20.71, 21.73, 17.49, 17.44, 18.14, 18.38, 18.09, 16.47, 14.76, 12.53, 11.62, 8.92, 7.24]
        # features = [49.0, 27.0, 2.5, 0.0, 0.0, 0.0, 84.0, 0.0, 88.0, 47.5, 41.5, 34.5, 36.5, 38.5, 38.0, 40.0, 37.0, 11.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 66.5, 52.0, 35.5, 0.0, 0.0, 0.0, 64.0, 0.0, 58.5, 23.5, 26.5, 22.0, 14.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 89.0, 68.5, 44.5, 0.0, 0.0, 0.0, 44.5, 0.0, 49.5, 0.5, 1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.5, 31.5, 59.5, 0.0, 0.0, 0.0, 77.0, 0.0, 74.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 85.0, 97.5, 115.5, 0.0, 0.0, 0.0, 117.0, 0.0, 113.0, 28.0, 33.0, 29.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 115.5, 121.5, 153.5, 0.0, 0.0, 6.5, 136.0, 15.0, 56.5, 6.0, 9.5, 5.0, 0.0, 0.0, 0.0, 7.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.5, 106.5, 108.0, 98.0, 0.0, 0.0, 88.0, 109.0, 0.0, 50.5, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5]
        # features = [0.07, 0.0, 0.0, 18.8, 7.33, 0.0, 4.13, 2.6, 0.0, 1.87, 24.0, 28.27, 34.6, 44.33, 53.87, 66.2, 79.2, 89.67, 94.53, 93.4, 86.8, 0.0, 0.0, 75.27, 65.33, 0.0, 1.4, 55.2, 39.87, 0.0, 28.67, 21.93, 0.0, 9.6, 44.2, 44.07, 46.33, 46.87, 53.8, 64.2, 71.47, 82.47, 95.6, 118.53, 141.07, 163.0, 184.0, 198.07, 210.47, 222.93, 223.93, 235.33, 241.2, 252.07, 263.73, 300.33, 308.47, 294.93, 282.2, 317.6, 341.73, 369.07, 406.27, 448.87, 492.8, 576.4, 648.0, 698.87, 739.33, 763.93, 769.13, 744.13, 710.13, 658.53, 620.87, 575.27, 499.73, 394.13, 340.93, 355.4, 270.73, 375.07, 364.87, 358.53, 338.53, 310.67, 298.07, 281.53, 263.27, 242.8, 231.2, 219.6, 172.53, 85.33, 67.87, 42.67, 23.87, 14.0, 25.13, 32.13, 16.4, 5.0, 5.87, 171.73, 157.33, 146.6, 141.27, 137.6, 101.8, 19.27, 0.0, 0.0, 0.0, 0.67, 0.33, 0.0, 0.0, 0.0, 0.0, 255.27, 0.07, 0.0, 97.0, 304.0, 234.87, 0.0, 308.53, 162.07, 0.33, 5.27, 6.2, 6.27, 4.53, 0.4, 0.0, 0.07, 233.27, 207.6, 185.53, 173.67, 169.93, 3.53, 29.13, 200.4, 14.53, 0.0, 229.47, 204.87, 28.93, 0.0, 63.87, 51.73, 0.0, 354.47, 222.07, 227.73, 208.27, 38.67, 0.0, 0.0, 23.67, 172.67, 161.4, 150.6, 146.53, 0.0, 140.47, 157.8]

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
        model = keras.models.load_model('CNN')
        y_pred_cnn = np.argmax(model.predict([np.array([features, ]), np.array([features, ]), np.array([features, ])]), axis=1)

        model = pickle.load(open('KNeighborsClassifier', 'rb'))
        y_pred_knn = model.predict(np.array([features, ]))

        model = pickle.load(open('KNN_DTW', 'rb'))
        y_pred_knn_dtw, _ = model.predict(np.array([features, ]))

        print("[CNN] Class: ", y_pred_cnn[0])
        print("[KNeighborsClassifier] Class: ", int(y_pred_knn[0]))
        print("[KNN_DTW] Class: ", int(y_pred_knn_dtw[0]))

        print("End Prediction!")

        print("Creating NetCDF...")
        src = cfg.get('variables', 'URL') + _datetime[0:4] + "/" + _datetime[4:6] + "/" + _datetime[6:8] + \
              "/wcm3_d03_" + _datetime + ".nc"
        dst = "test/wcm3_d03_20211008Z1200_class.nc.nc4"

        # Open the NetCDF file
        ncsrcfile = Dataset(src)

        ncols = len(long)
        nrows = len(lat)

        conc = ncsrcfile.variables["conc"][:]
        sfconc = ncsrcfile.variables["sfconc"][:]

        index = [i for i, _ in enumerate(areas) if
                 str(_['properties']['CODICE']) == str(code)][0]

        bbox = areas[index]['bbox']
        min_long = bbox[0]
        min_lat = bbox[1]
        max_long = bbox[2]
        max_lat = bbox[3]

        coordinates = areas[index]['geometry']['coordinates'][0][0]

        area_poly = Polygon(coordinates)

        index_min_lat, index_max_lat, index_min_long, index_max_long = get_index_lat_long(lat, long,
                                                                                          delta_lat, delta_long,
                                                                                          min_lat, max_lat,
                                                                                          min_long, max_long)

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
        concVar[:] = conc
        sfconcVar[:] = sfconc

        for i in range(index_min_lat, index_max_lat):
            for j in range(index_min_long, index_max_long):
                point = Point(long[j], lat[i])
                if point.within(area_poly):
                    classVar[0, i, j] = y_pred_knn + 1

        ncdstfile.close()

    else:
        print("Usage: datetime area_code")