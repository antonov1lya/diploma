import numpy as np
from numpy.random import choice
from tests import trivariate_umpu
from tests import umpu2
from tests import partial_pearson


def experiment(distribution, n, k, alpha):
    x = [i for i in range(8)]
    proba = [i[1] for i in distribution]
    s_ump = 0
    s_ump2 = 0
    s_partial = 0
    for _ in range(k):
        X = choice(x, p=proba, size=n)
        X = np.array([distribution[i][0] for i in X])
        s_ump += trivariate_umpu(X, alpha)
        s_ump2 += umpu2(X, alpha)
        s_partial += partial_pearson(X, alpha)
    return s_ump/k, s_ump2/k, s_partial/k
