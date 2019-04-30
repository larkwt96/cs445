import unittest
from esn import EchoStateNetwork
import matplotlib.pyplot as plt
import numpy as np


class TestEsn(unittest.TestCase):
    def testSin(self):
        total = 10
        bad = 0
        for _ in range(total):
            rmse = self.runSin()
            if rmse > 1:
                bad += 1
        if bad / total > .75:
            raise Exception(f'Too many failed: {int(100*bad/total)}%')

    def runSin(self):
        total_pts = 3000
        num_pts_train = int(.8*total_pts)
        self.ds = np.sin(np.arange(0, total_pts, .25)[:total_pts])[:, None]

        self.ds_train = self.ds[:num_pts_train, :]

        self.K = 0
        self.L = self.ds.shape[1]

        self.T0 = 100
        self.N = 20
        self.alpha = 0.99

        esn = EchoStateNetwork(self.K, self.N, self.L,
                               T0=self.T0, alpha=self.alpha)
        # test fit
        pre_ys = esn.predict(self.ds_train, Tf=total_pts)
        pre_rmse = np.sqrt(np.sum((pre_ys - self.ds)**2))
        esn.fit(self.ds_train)
        # test predict
        ys = esn.predict(self.ds_train, Tf=total_pts)
        #plt.plot(self.ds, label='ds')
        #plt.plot(ys, label='ys')
        # plt.show(True)
        # test rmse
        rmse = np.sqrt(np.sum((ys - self.ds)**2))
        return rmse

    def test_paper_exp(self):
        total = 10
        bad = 0
        for _ in range(total):
            pre_rmse, rmse = self.run_paper_exp()
            if rmse > 1:
                bad += 1
        if bad / total > 0.75:
            raise Exception(f'Too many failed: {int(100*bad/total)}%')

    def run_paper_exp(self):
        esn = EchoStateNetwork(0, 20, 1, T0=100,
                               alpha=.99)
        n = np.arange(1, 351)
        ds = (np.sin(n/4)/2).reshape(-1, 1)
        ys = esn.predict(ds[:100], Tf=350)
        pre_rmse = np.sqrt(np.sum((ys - ds)**2))
        esn.fit(ds[:300])

        # train measure
        ys = esn.predict(ds[:100], Tf=350)
        rmse = np.sqrt(np.sum((ys - ds)**2))
        #plt.plot(ds-ys, label='true')
        #plt.plot(ys, label='pred')
        # plt.legend()
        # plt.show(True)
        #self.assertGreater(pre_rmse, rmse)
        return pre_rmse, rmse

    def test_multi_dim_test(self):
        rmse_pattern = self.run_multi_dim_test(random=False)
        rmse_random = self.run_multi_dim_test(random=True)
        self.assertGreater(rmse_random, rmse_pattern)
        #print(rmse_random, rmse_pattern)

    def run_multi_dim_test(self, random=False):
        num_pts = 5000
        split = 0.9
        num_pts_train = int(split * num_pts)
        self.us = np.random.rand(num_pts, 3)
        self.ds = np.mean(self.us, axis=1)
        self.ds[1:] += self.ds[:-1]
        self.ds = self.ds[:, None]
        if random:
            self.ds = np.random.rand(*self.ds.shape)

        self.us_train = self.us[:num_pts_train, :]
        self.us_test = self.us[num_pts_train:, :]
        self.ds_train = self.ds[:num_pts_train, :]
        self.ds_test = self.ds[num_pts_train:, :]

        self.K = self.us.shape[1]
        self.L = self.ds.shape[1]

        self.T0 = 50
        self.N = 50
        self.alpha = 0.8

        esn = EchoStateNetwork(self.K, self.N, self.L,
                               T0=self.T0, alpha=self.alpha)
        # test fit
        esn.fit(self.ds_train, self.us_train)
        # test predict
        ys = esn.predict(self.ds_train, self.us)
        # test rmse
        rmse = np.sqrt(np.sum((ys - self.ds)**2))
        return rmse
