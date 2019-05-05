import os
import wfdb
import re
import numpy as np
import matplotlib.pyplot as plt

data_loc = os.path.join('data', 'dataset')


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


def plot_record(record, split=.05):
    ts_data = record['data']
    dims = record['sig_name']
    portion = int(split*ts_data.shape[0])
    freq = record['fs']
    t = np.arange(portion) / freq
    fig, axs = plt.subplots(8, 2, figsize=(20, 20))
    for i in range(15):
        y, x = i//2, i % 2
        axs[y, x].plot(t, ts_data[:portion, i])
        axs[y, x].set_ylabel(dims[i])
    plt.tight_layout()


def reformat_data(record):
    ds = record['data']
    us = np.array([record['labels']['vector'][:2] for _ in ds])
    return ds, us


def rmse(ys, ds):
    return np.mean((ys - ds)**2, axis=0)


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
