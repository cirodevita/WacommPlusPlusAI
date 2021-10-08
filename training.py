import collections
import sys

import pickle
import numpy as np
import matplotlib.pyplot as plt
from clang.cindex import xrange
from keras import Model
from numpy import shape
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from imblearn.over_sampling import SMOTE

training = 0

class KnnDtw(object):
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

    def fit(self, x, l):
        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in xrange(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in xrange(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in xrange(1, M):
            for j in xrange(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

    def _dist_matrix(self, x, y):
        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if np.array_equal(x, y):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            p = ProgressBar(shape(dm)[0])

            for i in xrange(0, x_s[0] - 1):
                for j in xrange(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])

                    dm_count += 1
                    p.animate(dm_count)

            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))
            dm_size = x_s[0] * y_s[0]

            p = ProgressBar(dm_size)

            for i in xrange(0, x_s[0]):
                for j in xrange(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)

            return dm

    def predict(self, x):
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()


class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        self.animate = self.animate_ipython

    def animate_ipython(self, iter):
        print('\r', self, sys.stdout.flush())
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)


def print_confusion_matrix(y_pred, y_test, labels, mode):
    conf_mat = confusion_matrix(y_pred, y_test)

    fig = plt.figure()
    fig.set_size_inches(7.0, 7.0)

    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c > 0:
                plt.text(j - .2, i + .1, c, fontsize=16)

    fig.colorbar(res)
    plt.title('Confusion Matrix ' + mode)
    _ = plt.xticks(range(3), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(3), [l for l in labels.values()])
    plt.show()


if training:
    hours = 168

    dataset = np.genfromtxt('dataset/dataset.csv', delimiter=';', skip_header=True)

    x = dataset[:, :hours]
    y = dataset[:, -1]

    y_0 = int(collections.Counter(y)[0.0])

    smt = SMOTE(sampling_strategy={1: int(y_0 / 2), 2: int(y_0 / 2)})
    x, y = smt.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)

    labels = {0: '0-67', 1: '67-230', 2: '230-4600'}
    # 4: '4600-46000', 5: '>46000'}

    # Baseline 2-KNN
    model_knn = KNeighborsClassifier(n_neighbors=2)
    model_knn.fit(x_train, y_train)
    # knnPickle = open('KNeighborsClassifier', 'wb')
    # pickle.dump(model_knn, knnPickle)
    y_pred = model_knn.predict(x_test)
    print("KNeighborsClassifier")
    print(classification_report(y_pred, y_test, target_names=[l for l in labels.values()]))
    print_confusion_matrix(y_pred, y_test, labels, "KNeighborsClassifier")

    # CNN
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    n_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes=n_classes)
    y_test = to_categorical(y_test, num_classes=n_classes)

    inputs1 = keras.layers.Input(x_train.shape[1:])
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs1)
    drop1 = keras.layers.Dropout(0.5)(conv1)
    pool1 = keras.layers.MaxPooling1D(pool_size=2)(drop1)
    flat1 = keras.layers.Flatten()(pool1)

    inputs2 = keras.layers.Input(x_train.shape[1:])
    conv2 = keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(inputs2)
    drop2 = keras.layers.Dropout(0.5)(conv2)
    pool2 = keras.layers.MaxPooling1D(pool_size=2)(drop2)
    flat2 = keras.layers.Flatten()(pool2)

    inputs3 = keras.layers.Input(x_train.shape[1:])
    conv3 = keras.layers.Conv1D(filters=64, kernel_size=11, activation='relu')(inputs3)
    drop3 = keras.layers.Dropout(0.5)(conv3)
    pool3 = keras.layers.MaxPooling1D(pool_size=2)(drop3)
    flat3 = keras.layers.Flatten()(pool3)

    merged = keras.layers.concatenate([flat1, flat2, flat3])

    dense1 = keras.layers.Dense(100, activation='relu')(merged)
    outputs = keras.layers.Dense(n_classes, activation='softmax')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # plot_model(model, to_file='model_cnn.png', show_shapes=True, show_layer_names=True)

    history = model.fit([x_train, x_train, x_train], y_train, epochs=60, batch_size=hours, validation_split=0.2)
    # model.save("CNN")
    test_loss, test_acc = model.evaluate([x_test, x_test, x_test], y_test, verbose=2)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict([x_test, x_test, x_test]), axis=1)
    print("CNN")
    print(classification_report(y_pred, y_test, target_names=[l for l in labels.values()]))
    print_confusion_matrix(y_pred, y_test, labels, "CNN")


    # KNN + DTW
    model_knn_dtw = KnnDtw(n_neighbors=2, max_warping_window=5)
    model_knn_dtw.fit(x_train, y_train)
    # with open("KNN_DTW.pkl", "wb") as knndtwPickle:
    #    pickle.dump(model_knn_dtw, knndtwPickle)
    y_pred, _ = model_knn_dtw.predict(x_test)
    print("KNN + DTW")
    print(classification_report(y_pred, y_test, target_names=[l for l in labels.values()]))
    print_confusion_matrix(y_pred, y_test, labels, "KNN + DTW")

"""
# CREATE CSV FILE
import json
import numpy as np
import pandas as pd

f = open('dataset/dataset.json')
data = json.load(f)
f.close()

lines = pd.read_csv("files/analisi_completa.csv", delimiter=';')

features = []
labels = []
labels_values = []

hours = 72

for d in data:
    for line in lines.iterrows():
        if d['id'].replace("/10:00", "") == line[1]['id'] and line[1]['COERENZA'] == "SI":
            if "NaN" not in d['features']:
                features.append(d['features'][0:hours])
                if 0.0 <= float(d['label']) <= 67.0:
                    labels.append(0)
                elif 67.0 < float(d['label']) <= 230.0:
                    labels.append(1)
                elif 230.0 < float(d['label']) <= 4600.0:
                    labels.append(2)
                elif 4600.0 < float(d['label']) <= 46000.0:
                    labels.append(3)
                else:
                    labels.append(4)

features = np.array(features, float)

x = pd.DataFrame(features)
x.columns = ["Feature " + str(i) for i in range(0, len(x.columns))]
y = pd.DataFrame(labels)
y.columns = ["Label (class)"]

dataframe = pd.concat([x, y], axis=1)
dataframe.to_csv("dataset/dataset.csv", sep=";", index=False)
"""