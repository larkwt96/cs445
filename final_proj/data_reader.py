import os
import wfdb
import re
import numpy as np

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


def vectorize_label(label):
    if label_has_na(label):
        return None
    age = norm_age(int(label['age']))
    sex = int(label['sex'] == 'male')
    status = int(label['status'] == 'Healthy control')
    return np.array([age, sex, status])


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
