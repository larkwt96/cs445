import os
from data_reader import *
from random import shuffle
import numpy as np

if __name__ == "__main__":
    records = get_records()
    data = [get_data(record) for record in records[:3]]
    print(data[0]['labels']['vector'])
    print(get_data(records[0])['data'].shape)
