from timeit import default_timer as timer
import numpy as np
from numpy import dot as npdot
import time
import matplotlib.pyplot as plt
import numexpr as ne
import sys


def sig(v, numexpr=False):
    if numexpr:
        return ne.evaluate( "1/(1 + exp(-v))")
    else:
        return 1/(1 + np.exp(-v))

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class CRBM:
    """
    Conditional Restricted Boltzmann Machine model.

    This class implements the CRBM model described in (1)



    ##### References

    (1): http://www.cs.toronto.edu/~fritz/absps/uai_crbms.pdf

    """
    def __init__(self, n_vis, n_hid, n_cond, seed=42, sigma=0.3, monitor_time=True):

        self.previous_xneg = None
        np.random.seed(seed)

        W = np.random.normal(0, sigma, [n_vis, n_hid])   # vis to hid
        A = np.random.normal(0, sigma, [n_vis, n_cond])  # cond to vis
        B = np.random.normal(0, sigma, [n_vis, n_cond])  # cond to hid

        v_bias = np.zeros([n_vis, 1]) 
        h_bias = np.zeros([n_hid, 1])

        dy_v_bias = np.zeros([n_vis, 1])
        dy_h_bias = np.zeros([n_hid, 1])

        self.W = np.array(W, dtype='float32')
        self.A = np.array(A, dtype='float32')
        self.B = np.array(B, dtype='float32')

        self.n_vis = n_vis
        self.n_hid = n_hid

        self.num_epochs_trained = 0
        self.lr = 0
        self.monitor_time = monitor_time



