import json
import pandas as pd
import numpy as np

from configparser import ConfigParser
cfg = ConfigParser()
cfg.read('config.ini')


def create_dataset(features, labels, file):
    features = np.array(features, float)

    x = pd.DataFrame(features)
    x.columns = ["Feature " + str(i) for i in range(0, len(x.columns))]
    y = pd.DataFrame(labels)
    y.columns = ["Label (class)"]

    dataframe = pd.concat([x, y], axis=1)
    dataframe.to_csv(cfg.get('training', file) + "_" + column.split("_")[1] + ".csv", sep=";", index=False)


f = open(cfg.get('training', 'DATASET_FILE_JSON'))
data = json.load(f)
f.close()

lines = pd.read_csv(open(cfg.get('training', 'ANALYSIS_FILE')), delimiter=',')

features_training = []
labels_training = []
features_test = []
labels_test = []

years = ["2019", "2020", "2021"]

hours = cfg.getint('training', 'HOURS')
column = cfg.get('training', 'COHERENCE_COLUMN_NAME')

for d in data:
    for line in lines.iterrows():
        id = str(line[1]['CODICE SITO']) + '_' + str(line[1]['DATA PRELIEVO'])
        if d['id'].replace("/10:00", "") == id and line[1][column] == "SI":
            if "NaN" not in d['features']:
                if d["year"] in years:
                    features_training.append(d['features'][0:hours])
                    if 0.0 <= float(d['label']) <= 67.0:
                        labels_training.append(0)
                    elif 67.0 < float(d['label']) <= 230.0:
                        labels_training.append(1)
                    # elif 230.0 < float(d['label']) <= 4600.0:
                    #    labels.append(2)
                    else:
                        labels_training.append(2)
                else:
                    features_test.append(d['features'][0:hours])
                    if 0.0 <= float(d['label']) <= 67.0:
                        labels_test.append(0)
                    elif 67.0 < float(d['label']) <= 230.0:
                        labels_test.append(1)
                    # elif 230.0 < float(d['label']) <= 4600.0:
                    #    labels.append(2)
                    else:
                        labels_test.append(2)

# TRAINING SET
create_dataset(features_training, labels_training, "DATASET_TRAINING_FILE_CSV")
# TEST SET
# create_dataset(features_test, labels_test, "DATASET_TEST_FILE_CSV")
