import os
import re
from copy import deepcopy
from itertools import product
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
import scipy.signal
import sklearn.metrics
import wfdb

from esn import EchoStateNetwork

data_loc = os.path.join('data', 'dataset')


def get_signal(record, dim=0, keep=1, denoise=True, norm=True, min_len=None):
    T = record['fs']
    signal = record['data'][:, dim]
    if min_len is not None and signal.shape[0] < min_len:
        return None, T
    if norm:
        signal = baseline(signal)
    if denoise:
        signal = reduce_noise(signal)
    if min_len is not None and signal.shape[0] > min_len:
        beg = int(signal.shape[0]/2 - min_len/2)
        signal = signal[beg:beg+min_len]
    if keep < 1:
        length = int(signal.shape[0]*(1-keep))
        signal = signal[length:-length]
    return signal, T


def fft_exp(signal, T, distance=25):
    sf = np.fft.rfft(signal)
    sf = np.abs(sf)
    sf = np.log10(sf**2)
    xf = np.fft.rfftfreq(signal.shape[0], d=1/T)
    peaks, _ = scipy.signal.find_peaks(sf, distance=distance)
    peaks = peaks[np.argsort(-sf[peaks])]
    return xf, sf, peaks


def get_feat(record, classi=-1, freqs=None, high_freq=25, sf_range=75):
    signal, T = get_signal(record, min_len=115200)
    if signal is None:
        return None
    t = np.arange(signal.shape[0])/T
    xf, sf, peaks = fft_exp(signal, T)
    peaks = peaks[xf[peaks] < high_freq]  # get low freq
    if freqs is None:
        return np.hstack((xf[peaks][:sf_range], sf[peaks][:sf_range], [classi]))
    elif peaks.shape[0] < freqs:
        return None
    else:
        peaks = peaks[:freqs]  # get tallest
        return np.hstack((xf[peaks], sf[peaks], [classi]))


def get_feats(records, classi, sub=None):
    feats = None
    if sub is None:
        sub = len(records)
    for record in records:
        feat = get_feat(record, classi=classi)
        if feat is None:
            continue
        if feats is None:
            feats = feat[None, :]
        else:
            feats = np.vstack((feats, feat))
        if feats.shape[0] == sub:
            break
    return feats


def split_v(V, split):
    split_ind = int(split * V.shape[0])
    if split_ind == V.shape[0]:
        split_ind = max(0, V.shape[0])
    return V[:split_ind], V[split_ind:]


def get_xt(data_sets, sub=None, split=.8, usub=None):
    print('Getting healthy features')
    healthy_feat = get_feats(data_sets[0], 0, sub=sub)
    np.random.shuffle(healthy_feat)
    train_hf, test_hf = split_v(healthy_feat, split)

    print('Getting sick features')
    unhealthy_feat = get_feats(data_sets[1], 1, sub=sub)
    np.random.shuffle(unhealthy_feat)
    train_uf, test_uf = split_v(unhealthy_feat, split)

    train_v = np.vstack((train_hf, train_uf))
    np.random.shuffle(train_v)
    test_v = np.vstack((test_hf, test_uf))
    np.random.shuffle(test_v)

    train_x, train_t = train_v[:, :-1], train_v[:, -1:]
    test_x, test_t = test_v[:, :-1], test_v[:, -1:]
    return train_x, train_t, test_x, test_t


def plot_peaks(xf, sf, peaks, xlim=(0, 10)):
    plt.figure()
    plt.plot(xf, sf)
    plt.plot(xf[peaks], sf[peaks], 'x')
    plt.xlim(xlim)


def plot_cm(cm):
    plt.figure(figsize=(2, 2))
    plt.imshow(1 - cm / np.sum(cm, axis=1)
               [:, None], cmap='gray', vmax=1, vmin=0)
    plt.xlabel('Predicted')
    plt.xticks([0, 1])
    plt.ylabel('Actual')
    plt.yticks([0, 1])
    plt.colorbar()


def bsr(cm):
    return np.trace(cm / np.sum(cm, axis=1)[:, None]) / cm.shape[0]


def acc(cm):
    return 1-np.sum(cm - np.identity(cm.shape[0])*np.diagonal(cm)) / np.sum(cm)


def get_records():
    """ patient as returned by get_patients """
    records_f = open(os.path.join(data_loc, 'RECORDS'))
    return [line.strip() for line in records_f]


def get_data(record):
    """
    patient as returned by get_patients; record as returned by get_records
    """
    data = wfdb.rdsamp(os.path.join(data_loc, record))
    labels = pull_labels(data)
    data[1]['labels'] = labels
    data[1]['data'] = data[0]
    return data[1]


def get_all_data(verbose=False):
    """
    Probably shouldn't be running this, and it may take a while
    """
    data = {}
    for record in get_records():
        data[record] = get_data(record)
    return data


def label_has_na(label):
    for key in label:
        if label[key] == 'n/a':
            return True
    return False


def norm_age(age):
    return age / 100


def plot_record(record=None, split=.05, raw=None):
    ts_data = record['data'] if raw is None else raw
    dims = record['sig_name']
    freq = record['fs']
    portion = int(split*ts_data.shape[0])
    t = np.arange(portion) / freq
    fig, axs = plt.subplots(8, 2, figsize=(20, 20))
    for i in range(15):
        if ts_data.shape[1] == i:
            break
        y, x = i//2, i % 2
        axs[y, x].plot(t, ts_data[-portion:, i])
        axs[y, x].set_ylabel(dims[i])
    plt.tight_layout()


def reformat_data(record):
    ds = record['data']
    us = np.array([record['labels']['vector'][:2] for _ in ds])
    return ds, us


def calc_rmse(ys, ds):
    return np.mean((ys - ds)**2, axis=0)


def plot_record(record=None, split=.05, raw=None, freq=None):
    if raw is None:
        ts_data = record['data']
    else:
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        ts_data = raw
    rows = ts_data.shape[1] // 2 + 1
    height = int(20/8*rows)
    dims = record['sig_name']
    freq = record['fs'] if freq is None else record['fs']//freq
    portion = int(split*ts_data.shape[0])
    t = np.arange(portion) / freq

    fig, axs = plt.subplots(rows, 2, figsize=(20, height))
    if axs.ndim == 1:
        axs = axs.reshape(-1, 1)
    for i in range(15):
        if ts_data.shape[1] == i:
            break
        y, x = i//2, i % 2
        axs[y, x].plot(t, ts_data[-portion:, i])
        axs[y, x].set_ylabel(dims[i])
    plt.tight_layout()


def diff_data(ds):
    d0 = ds[0]
    return d0, ds[1:] - ds[:-1]


def undiff_data(d0, ds):
    rebuild = [d0]
    for d in ds:
        rebuild.append(rebuild[-1] + d)
    return np.array(rebuild)


def baseline(signal, freq=1000):
    """
    signal can be multiple dims
    freq is the number of samples per second
    """
    new_signal = np.copy(signal)
    L = signal.shape[0]
    w = 2*(freq // 4)  # even number roughly half a second
    for i in range(L):
        low = max(i - w//2, 0)
        high = min(i + w//2, L)
        new_signal[i] = signal[i] - np.mean(signal[low:high], axis=0)
    return new_signal


def reduce_noise_multi_dim(signal, level, threshold, wavelet):
    res = np.zeros_like(signal)
    for dim in range(signal.shape[1]):
        sig = signal[:, dim]
        res[:, dim] = reduce_noise(sig, level, threshold, wavelet)
    return res


def reduce_noise(signal, level=None, threshold=.2, wavelet='sym5'):
    """
    signal is np ndarray
    level is the level of decompositions to denoise to
    threshold is the threshold percentage
    """
    if signal.ndim > 1:
        return reduce_noise_multi_dim(signal, level, threshold, wavelet)

    threshold *= (np.mean(np.abs(signal)) + np.max(np.abs(signal))) / 2
    appx, *coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs = [pywt.threshold(coeff, value=threshold, mode="soft")
              for coeff in coeffs]
    return pywt.waverec([appx, *coeffs], wavelet)[:signal.shape[0]]


def vectorize_label(label):
    if label_has_na(label):
        return None
    age = norm_age(int(label['age']))
    sex = int(label['sex'] == 'male')
    status = int(label['status'] == 'Healthy control')
    return np.array([age, sex, status])


def prepare_record(self, record):
    ds = record['data']


def get_totals(data):
    ages = []
    genders = []
    stats = []
    for record in data:
        labels = record['labels']
        ages.append(labels['age'])
        genders.append(labels['sex'])
        stats.append(labels['status'])
    ages = np.unique(ages)
    genders = np.unique(genders)
    stats = np.unique(stats)
    return ages, genders, stats


def proc_comment(comment):
    label = [part.strip() for part in comment.split(':', 1)]
    if len(label) == 1:
        label += ['n/a']
    elif label[1] == '':
        label[1] = 'n/a'
    return label


def pull_labels(data):
    comments = data[1]['comments']
    keep = {'age': 'age', 'sex': 'sex', 'Reason for admission': 'status'}
    labels = {}
    for comment in comments:
        label_name, label_value = proc_comment(comment)
        if label_name not in keep:
            continue
        labels[keep[label_name]] = label_value
    labels['vector'] = vectorize_label(labels)
    return labels
