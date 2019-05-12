import matplotlib.pyplot as plt
from data_reader import *
from random import shuffle
import numpy as np
from esn import EchoStateNetwork
from itertools import product
import pywt
from copy import deepcopy
import pickle

alpha_c = "\u03b1"
np.set_printoptions(precision=2)


def reformat_data(record, step=1, diff=False, sel=(0, ), keep=1):
    # format data
    ds = record['data']
    if keep < 1:
        length = int(ds.shape[0]*(1-keep)/2)
        ds = ds[length:-length]
    # focus dims
    if sel is not None:
        ds = ds[:, sel]
    # baseline
    ds = baseline(ds)
    # denoise
    for i in range(ds.shape[1]):
        ds[:, i] = reduce_noise(ds[:, i])
    if step > 1:
        ds = ds[:ds.shape[0] - ds.shape[0] % step]
        ds = ds.reshape(ds.shape[0]//step, step, ds.shape[1])
        ds = np.mean(ds, axis=1).reshape(-1, ds.shape[2])
    # diff
    if diff:
        ds = diff_data(ds)  # no input data
        us = np.zeros((ds[1].shape[0], 0))
    else:
        us = np.zeros((ds.shape[0], 0))
    return ds, us


def experiment(ds, us, split_ind, T0, N, alpha, noise, wback_fac):
    K = us.shape[1]
    L = ds.shape[1]
    esn = EchoStateNetwork(K, N, L, T0, alpha, noise=noise)
    esn.Wback *= wback_fac
    esn.fit(ds[:split_ind], us[:split_ind])
    return esn.predict(ds[:split_ind], us), esn


if __name__ == "__main__":
    records = get_records()
    orig_data = [get_data(record) for record in records]
    # remove data missing labels
    data = [record for record in orig_data if record['labels']['status'] != "n/a"]
    print(f"Number of records {len(data)}")
    print("Shape of data: {}".format(data[0]['data'].shape))
    healthy_data = [record for record in data if record['labels']
                    ['status'] == 'Healthy control']
    print(f"Number of controls: {len(healthy_data)}")
    unhealthy_data = [
        record for record in data if record['labels']['status'] != 'Healthy control']
    print(f"Number of sick: {len(unhealthy_data)}")

    Ns = [10, 15, 20, 25, 50, 75, 100, 125, 150, 300, 500, 1000]
    alphas = [.8, .85, .9, .925, 0.95, .98, 1, 2, 5, 7]
    steps = [1, 2, 5, 10]
    wback_facs = [1, .2, .5, 1.3]

    T0 = 5000  # more than enough
    N = 1000
    alpha = .98
    wback_fac = 1
    noise = None

    sel = (0, )
    diff = False
    step = 1
    keep = 1
    split = 0.9
    record = healthy_data[0]

    T0 = T0 // step
    ds, us = reformat_data(record, step, diff, sel, keep)
    if diff:
        d0, ds = ds
    split_ind = int(ds.shape[0]*split)
    print(ds.shape)
    ys, esn = experiment(ds, us, split_ind, T0, N, alpha, noise, wback_fac)
    rmse = calc_rmse(ds[split_ind:], ys[split_ind:])
    print(rmse)
    if diff:
        ys = undiff_data(d0, ys)
        ds = undiff_data(d0, ds)

    inter = 750//step
    plt.figure(figsize=(30, 10))
    bot = max(split_ind-inter, 0)
    top = min(split_ind+inter, ys.shape[0])
    t = np.arange(bot, top) - split_ind
    plt.plot(t, ds[bot:top, 0], '--', label='actual')
    plt.plot(t, ys[bot:top, 0], label='appx')
    if diff:
        pass
    else:
        plt.ylim((-.5, .75))
    plt.legend()

    plt.savefig('./output.png')
    pickle.dump((ds, ys, esn.W, esn.Win, esn.Wout,
                 esn.Wback), open('./state.p', 'wb'))
    plt.show()
