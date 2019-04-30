import os
from data_reader import *
from random import shuffle
import numpy as np

if __name__ == "__main__":
    records = get_records()
    print(get_data(records[0])['data'].shape)


def get_totals():
    shuffle(records)
    ages = []
    genders = []
    stats = []
    for record in records[:50]:
        label = get_data(record)['labels']
        ages.append(label['age'])
        genders.append(label['sex'])
        stats.append(label['status'])
    ages = np.unique(ages)
    genders = np.unique(genders)
    stats = np.unique(stats)
    print(ages, len(ages))
    print(genders, len(genders))
    print(stats, len(stats))
