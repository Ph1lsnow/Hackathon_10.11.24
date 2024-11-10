import os.path
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import pyedflib


@keras.saving.register_keras_serializable(package="my_package", name="f1_macro")
def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision_0 = tp / (tp + fp + K.epsilon())
    recall_0 = tp / (tp + fn + K.epsilon())
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0 + K.epsilon())
    f1_0 = tf.where(tf.math.is_nan(f1_0), tf.zeros_like(f1_0), f1_0)

    precision_1 = tn / (tn + fn + K.epsilon())
    recall_1 = tn / (tn + fp + K.epsilon())
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1 + K.epsilon())
    f1_1 = tf.where(tf.math.is_nan(f1_1), tf.zeros_like(f1_1), f1_1)

    # Macro-average F1-score
    f1_macro = (f1_0 + f1_1) / 2

    return f1_macro


@keras.saving.register_keras_serializable(package="my_package", name="f1_score")
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Округление предсказаний до ближайшего целого (0 или 1)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def read_edf_signals_and_markers(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signal_length = f.getNSamples()[0]
    sample_frequency = f.getSampleFrequency(0)

    # Initialize an empty array to store all data
    data = np.zeros((n + 1, signal_length))

    # Create a time array
    time = np.arange(signal_length) / sample_frequency
    data[0, :] = time

    # Read signals and store them in the data array
    for i in range(n):
        data[i + 1, :] = f.readSignal(i)

    f._close()
    del f

    return data.transpose()


def create_chanks(data: np.array, standart_len, step=60):
    chanks = []

    for start in range(0, len(data) - standart_len, step):
        chanks.append(data[start:start + standart_len, 1:4])
    return chanks


def classify_seconds(probabilities, time, threshold=0.5, min_dist_between=0.5, min_len=0.5):
    # Определяем класс для каждой секунды
    classes = np.argmax(probabilities, axis=1)

    # Инициализируем переменные для отслеживания текущего класса и отрезков времени
    current_class = None
    start_time_index = 0
    segments = []
    last_end = [-1, -1, -1]
    sum_of_score = 0

    # Проходим по всем секундам
    for cl in range(3):
        start_time_index = 0
        sum_of_score = 0
        for i in range(len(time)):

            if probabilities[i][cl] >= threshold: sum_of_score += probabilities[i][cl]

            # Если класс изменился или достигнут конец вектора
            if probabilities[i][cl] < threshold:

                # Если это не начало вектора, то завершаем текущий отрезок
                if i != 0 and i - 1 > start_time_index and min_dist_between + last_end[cl] <= time[start_time_index] and \
                        time[i - 1] >= min_len + time[start_time_index]:
                    segments.append([start_time_index, i - 1, cl, sum_of_score / (i - start_time_index)])
                    last_end[cl] = time[i - 1]

                elif i != 0 and i - 1 > start_time_index and min_dist_between + last_end[cl] > time[start_time_index]:
                    segments[-1][1] = i - 1
                    segments[-1][3] = (segments[-1][3] + (sum_of_score / (i - start_time_index))) / 2
                    last_end[cl] = time[i - 1]

                for q in range(start_time_index, i):
                    sum_of_score -= probabilities[q][cl]

                start_time_index = i + 1
    segments.sort(key=lambda x: x[0])
    return segments


def find_anomals(segments, time, probabilities, max_dist_between=1, min_delta_score=0.1):
    anomals = []
    for i in range(len(segments) - 1):
        if segments[i][2] == segments[i + 1][2] and time[segments[i + 1][0]]-time[segments[i][1]] <= max_dist_between:
            max_prob = max(probabilities[segments[i][1]][segments[i][2]],
                           probabilities[segments[i + 1][1]][segments[i + 1][2]])
            min_prob = 1
            for t in range(segments[i][1], segments[i + 1][1] + 1):
                min_prob = min(min_prob, probabilities[t][segments[i][2]])
            if max_prob - min_prob >= min_delta_score:
                anomals.append((segments[i][1], segments[i + 1][0], 4, max_prob - min_prob))
    return anomals


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def func_with_nn(path, threshold=0.5, min_dist_between=0.5, min_len=0.5, min_delta_score=0.1, max_dist_between=1):
    model_cl1 = load_model(resource_path('model/model_f1_81_cl1_v2.keras'))
    model_cl2 = load_model(resource_path('model/model_f1_75_cl2_v2.keras'))
    model_cl3 = load_model(resource_path('model/model_f1_macro_78_cl3_v2.keras'))

    data = read_edf_signals_and_markers(path)

    standart_deviation_in_chank = 30
    standart_len = 2 * standart_deviation_in_chank + 1

    step = 60
    chanks = np.array(create_chanks(data, standart_len, step=step))

    y_pred_cl1 = model_cl1.predict(chanks, verbose=0)
    y_pred_cl2 = model_cl2.predict(chanks, verbose=0)
    y_pred_cl3 = model_cl3.predict(chanks, verbose=0)

    time = []

    for start in range(0, len(data) - standart_len, step):
        time.append(data[start + standart_deviation_in_chank, 0])

    probabilities = [np.zeros(3) for i in range(len(y_pred_cl1))]

    for i in range(len(y_pred_cl1)):
        probabilities[i][0] = y_pred_cl1[i][1]
        probabilities[i][1] = y_pred_cl2[i][1]
        probabilities[i][2] = y_pred_cl3[i][1]

    classif = classify_seconds(probabilities, time, threshold=threshold, min_dist_between=min_dist_between,
                               min_len=min_len)

    anomals = find_anomals(classif, time, probabilities, max_dist_between=max_dist_between,
                           min_delta_score=min_delta_score)
    classif += anomals
    classif.sort(key=lambda x: x[0])

    out = []

    for s in classif:
        if s[2] == 0:
            out.append([time[s[0]], -1, 'ds1', s[3]])
            out.append([time[s[1]], -1, 'ds2', -1])
        elif s[2] == 1:
            out.append([time[s[0]], -1, 'is1', s[3]])
            out.append([time[s[1]], -1, 'is2', -1])
        elif s[2] == 2:
            out.append([time[s[0]], -1, 'swd1', s[3]])
            out.append([time[s[1]], -1, 'swd2', -1])
        elif s[2] == 4:
            out.append([time[s[0]], -1, 'an1', s[3]])
            out.append([time[s[1]], -1, 'an2', -1])

    return out

