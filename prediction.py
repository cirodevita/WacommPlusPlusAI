import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
from tensorflow import keras
from main import load_areas, get_lat_long, worker
# from training import KnnDtw
import matplotlib.pyplot as plt
from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('config.ini')


if __name__ == "__main__":
    if len(sys.argv) > 2:
        _datetime = sys.argv[1]
        code = sys.argv[2]

        # print("Loading areas...")
        # areas = load_areas()
        # print("Loading completed!")

        # print("Loading lat/long...")
        # lat, long, delta_lat, delta_long = get_lat_long()
        # print("Loading completed!")

        # print("Dataset creation...")
        # new_datetime = _datetime[6:8] + "/" + _datetime[4:6] + "/" + _datetime[2:4] + "/" + _datetime[9:11] + ":00"
        # measure = {"id": str(code) + "_" + new_datetime[:-6], "year": datetime[0:4], "datetime": new_datetime,
        # "code": code, "outcome": "--"}
        # dataset = worker(areas, lat, long, delta_lat, delta_long, measure)
        # print("Dataset created!")

        features = [12.25, 12.69, 12.03, 11.36, 10.25, 8.84, 7.54, 6.27, 5.75, 5.86, 6.05, 5.91, 6.03, 6.19, 6.52, 7.09, 7.72, 8.87, 10.09, 11.65, 12.55, 13.08, 13.46, 16.14, 19.91, 21.03, 20.7, 19.41, 18.28, 17.65, 17.75, 17.93, 17.35, 16.93, 16.6, 16.25, 15.44, 14.59, 14.1, 13.41, 12.28, 11.16, 10.11, 9.27, 8.05, 7.27, 7.0, 8.01, 8.42, 8.18, 8.47, 7.81, 6.99, 6.31, 5.83, 5.73, 6.13, 5.61, 8.24, 7.49, 7.98, 6.46, 6.84, 5.72, 5.57, 4.18, 2.26, 6.23, 5.73, 2.03, 0.43, 5.94, 5.06, 1.67, 5.95, 4.6, 3.21, 0.06, 11.85, 10.82, 10.65, 5.1, 6.01, 5.38, 7.09, 3.82, 9.49, 10.88, 10.18, 7.64, 2.66, 20.51, 19.38, 7.14, 0.08, 9.05, 19.64, 1.01, 42.07, 37.95, 35.28, 7.62, 39.49, 29.24, 31.36, 37.51, 38.46, 36.23, 34.17, 25.85, 26.69, 22.73, 19.87, 12.52, 1.45, 18.42, 13.46, 2.54, 0.02, 8.65, 14.57, 0.51, 24.29, 19.93, 15.61, 2.31, 22.02, 15.95, 14.84, 13.08, 14.24, 14.45, 16.02, 18.1, 20.74, 22.91, 25.29, 25.99, 24.79, 22.22, 17.96, 11.73, 9.86, 10.29, 10.49, 10.58, 10.43, 10.68, 11.92, 12.94, 16.37, 24.35, 23.5, 19.64, 19.36, 20.71, 21.73, 17.49, 17.44, 18.14, 18.38, 18.09, 16.47, 14.76, 12.53, 11.62, 8.92, 7.24]

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
        y_pred = np.argmax(model.predict([np.array([features, ]), np.array([features, ]), np.array([features, ])]), axis=1)
        # # model = joblib.load("CNN")
        # # y_pred = model.predict(np.array([features, ]))
        print("Class: ", y_pred)
    else:
        print("Usage: datetime area_code")
