import json

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

f = open('dataset/dataset_mean_2019.json')
data = json.load(f)
f.close()

"""
lines = pd.read_csv("files/analisi.csv", delimiter=';')

features = []
labels = []
id = []

for d in data:
    for line in lines.iterrows():
        if d['id'] == line[1]['NUMERO SCHEDA'] and (line[1]['DA CONSIDERARE']).upper() == "SI":
            if "NaN" not in d['features']:
                features.append(d['features'])
                labels.append(float(d['label']))
                id.append(d['id'])

for index, item in enumerate(features):
    for i, _item in enumerate(item):
        item[index] = float(_item)

x = pd.DataFrame(features)
x.columns = ["Feature " + str(i) for i in range(0, len(x.columns))]
y = pd.DataFrame(labels)
y.columns = ["Label"]
"""

"""
df = pd.read_csv("files/housing.csv")
df.dropna(inplace=True)
df = df.drop('ocean_proximity', axis=1)

x = df.drop('median_house_value', axis=1)
y = df['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential([
    keras.layers.Input(shape=(8,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ], name="MLP_model")

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
model.summary()

model.fit(x_train, y_train, epochs=400, validation_split=0.2, verbose=1)
y_prediction = model.predict(x_test)
print(y_prediction)
print(y_test)
score = r2_score(y_test, y_prediction)
print('r2 socre is ', score)
"""

"""
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
"""

"""
from datetime import datetime, timedelta

import numpy as np

from keras import Input, Model
from keras.layers import Embedding, Conv1D, BatchNormalization, Activation, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from main import cfg

# lines = pd.read_csv(cfg.get('files', 'MEASURES'), delimiter=';')

# for d in data:
#    date = ""
#    for line in lines.iterrows():
#        if d['id'] == line[1]['NUMERO SCHEDA']:
#            date = str(line[1]['DATA PRELIEVO']) + cfg.get('variables', 'SAMPLE_HOUR')
#    for idx, val in enumerate(d['features']):
#        if val == "NaN":
#            reference_hour = datetime.strptime(date, '%d/%m/%y/%H:%M') - timedelta(hours=idx+1)
#            if not datetime(day=5, month=7, year=2020, hour=0, minute=0) < reference_hour < datetime(day=6, month=7, year=2020, hour=0, minute=0) \
#                    and not datetime(day=18, month=8, year=2020, hour=0, minute=0) < reference_hour < datetime(day=19, month=8, year=2020, hour=0, minute=0) \
#                    and not datetime(day=9, month=9, year=2020, hour=0, minute=0) < reference_hour < datetime(day=10, month=9, year=2020, hour=0, minute=0) \
#                    and not datetime(day=6, month=10, year=2020, hour=0, minute=0) < reference_hour < datetime(day=12, month=10, year=2020, hour=0, minute=0):
#                print(d['id'], reference_hour)

# _id = pd.DataFrame(id)
# _id.columns = ["ID"]

# df = pd.concat([x, y], axis=1)
# df.to_csv('dataset/dataset_mean_2019.csv', index=False)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

LR = LinearRegression()
LR.fit(x_train, y_train)
y_prediction = LR.predict(x_test)

print(y_test)
print(y_prediction)

score = r2_score(y_test, y_prediction)
print('r2 socre is ', score)
# print('mean_sqrd_error is ==', mean_squared_error(y_test, y_prediction))
# print('root_mean_squared error of is ==', np.sqrt(mean_squared_error(y_test, y_prediction)))
"""