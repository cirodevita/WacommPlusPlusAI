import json
import pandas as pd
from sklearn.model_selection import train_test_split

f = open('dataset/dataset_mean_2019.json')
data = json.load(f)
f.close()

lines = pd.read_csv("files/analisi.csv", delimiter=';')

features = []
labels = []
id = []

for d in data:
    for line in lines.iterrows():
        if d['id'] == line[1]['NUMERO SCHEDA'] and (line[1]['DA CONSIDERARE']).upper() == "SI":
            if "NaN" not in d['features']:
                features.append(d['features'])
                labels.append(d['label'])
                id.append(d['id'])

x = pd.DataFrame(features)
x.columns = ["Feature " + str(i) for i in range(0, len(x.columns))]
y = pd.DataFrame(labels)
y.columns = ["Label"]

print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

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

main_input = Input(shape=(168,), name='main_input')
emb = Embedding(256*8, output_dim=64, input_length=168)(main_input)
conv1d = Conv1D(filters=32, kernel_size=3, padding='valid')(emb)
bn = BatchNormalization()(conv1d)
sgconv1d = Activation('sigmoid')(bn)
conv1d_2 = Conv1D(filters = 32, kernel_size = 3, padding = 'valid')(sgconv1d)
bn2 = BatchNormalization()(conv1d_2)
sgconv1d_2 = Activation('sigmoid')(bn2)
#conv = Multiply()([conv1d, sgconv1d])
#pool = MaxPooling1D(pool_size = 32)(conv)
out = Flatten()(sgconv1d_2)
out = Dense(512, activation = 'relu')(out)
out = Dense(256, activation = 'relu')(out)

loss = Dense(1, activation = 'linear')(out)

model = Model(inputs = [main_input], outputs = [loss])
model.compile(loss='mean_absolute_percentage_error', optimizer = 'Adam', \
              metrics=['mean_squared_error', 'mean_absolute_percentage_error'])

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