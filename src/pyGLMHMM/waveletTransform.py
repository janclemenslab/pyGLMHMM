import numpy as np

def _w_corr(x, y, w):
    return _w_cov(x, y, w) / np.sqrt(_w_cov(x, x, w) * _w_cov(y, y, w))

def _w_cov(x, y, w):
    return np.sum(w * (x - _w_mean(x, w)) * (y - _w_mean(y, w)), axis = 0) / np.sum(w, axis = 0)

def _w_mean(x, w):
    return np.sum(w * x, axis = 0) / np.sum(w, axis = 0)