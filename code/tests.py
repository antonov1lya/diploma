import numpy as np
from numpy import cov
from scipy.special import softmax
from find_thresholds import find_thresholds
from scipy.stats import t as student


def test(proba, value, u, alpha):
    C1, C2, gamma1, gamma2 = find_thresholds(proba, value, alpha)
    if C1 != C2:
        if u < C1 or u > C2:
            return 1
        if C1 < u < C2:
            return 0
        if u == C1:
            if np.random.random() < gamma1:
                return 1
            else:
                return 0
        if u == C2:
            if np.random.random() < gamma2:
                return 1
            else:
                return 0
    else:
        if u == C1:
            if np.random.random() < gamma1:
                return 1
            else:
                return 0
        else:
            return 1


def trivariate_umpu(x, alpha):
    t1 = np.sum(x[:, 0]*x[:, 1])
    t2 = np.sum(x[:, 0]*x[:, 2])
    t3 = np.sum(x[:, 1]*x[:, 2])
    t4 = np.sum(x[:, 0])
    t5 = np.sum(x[:, 1])
    t6 = np.sum(x[:, 2])

    n = x.shape[0]

    k1 = t1
    k2 = t2
    k3 = t3
    k4 = t4-t1-t2
    k5 = t5-t1-t3
    k6 = t6-t2-t3
    k7 = n+t1+t2+t3-t4-t5-t6

    f = [0]
    for i in range(1, n+1):
        f.append(f[-1] + np.log(i))

    value = []
    proba = []
    for u in range(n+1):
        a1 = 0 <= k1 - u <= n
        a2 = 0 <= k2 - u <= n
        a3 = 0 <= k3 - u <= n
        a4 = 0 <= k4 + u <= n
        a5 = 0 <= k5 + u <= n
        a6 = 0 <= k6 + u <= n
        a7 = 0 <= k7 - u <= n
        if a1 and a2 and a3 and a4 and a5 and a6 and a7:
            value.append(u)
            proba.append(-(f[u] + f[k1 - u] + f[k2 - u] + f[k3 - u] +
                         f[k4 + u] + f[k5 + u] + f[k6 + u] + f[k7 - u]))

    u = np.sum(x[:, 0]*x[:, 1]*x[:, 2])
    proba = softmax(proba)

    return test(proba, value, u, alpha)


def bivariate_umpu(x, alpha):
    t1 = np.sum(x[:, 0])
    t2 = np.sum(x[:, 1])

    n = x.shape[0]

    k1 = t1
    k2 = t2
    k3 = n-t1-t2

    f = [0]
    for i in range(1, n+1):
        f.append(f[-1] + np.log(i))

    value = []
    proba = []
    for u in range(n+1):
        a1 = 0 <= k1 - u <= n
        a2 = 0 <= k2 - u <= n
        a3 = 0 <= k3 + u <= n
        if a1 and a2 and a3:
            value.append(u)
            proba.append(-(f[u] + f[k1 - u] + f[k2 - u] + f[k3 + u]))

    proba = softmax(proba)
    u = np.sum(x[:, 0]*x[:, 1])

    return test(proba, value, u, alpha)


def umpu2(x, alpha):
    x0 = x[x[:,2]==0][:,[0,1]]
    x1 = x[x[:,2]==1][:,[0,1]]
    new_alpha = 1 - np.sqrt(1-alpha)
    if bivariate_umpu(x0, new_alpha)==1 or bivariate_umpu(x1, new_alpha)==1:
        return 1
    else:
        return 0

def partial_pearson(x, alpha):
    n = x.shape[0]
    s = cov(x.T, ddof=1)
    s_inv = np.linalg.inv(s)
    par = s_inv[0][1]/np.sqrt(s_inv[0][0] * s_inv[1][1])
    par_stat = np.sqrt(n-3) * (par / np.sqrt(1-np.power(par, 2)))
    C1 = student.ppf(alpha/2, n-3)
    C2 = student.ppf(1-alpha/2, n-3)
    if C1 < par_stat < C2:
        return 0
    else:
        return 1
