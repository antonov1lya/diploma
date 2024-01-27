import numpy as np


def det(a11, a12, a21, a22):
    return a11*a22-a12*a21


def solver(a11, a12, a21, a22, b1, b2):
    delta = det(a11, a12, a21, a22)
    delta1 = det(b1, a12, b2, a22)
    delta2 = det(a11, b1, a21, b2)
    return delta1/delta, delta2/delta


def find_thresholds(proba, value, alpha):
    E = np.dot(proba, value)
    prefix = np.insert(np.cumsum(proba), 0, 0)[:-1]
    suffix = np.append(np.cumsum(proba[::-1])[::-1], 0)[1:]
    prefix_E = np.insert(np.cumsum(proba * value), 0, 0)[:-1]
    suffix_E = np.append(np.cumsum((proba * value)[::-1])[::-1], 0)[1:]
    n = len(value)
    for i in range(n):
        for j in range(i+1, n):
            c1 = value[i]
            c2 = value[j]
            gamma1, gamma2 = solver(proba[i], proba[j], c1*proba[i], c2*proba[j],
                                    alpha-prefix[i]-suffix[j], alpha*E-prefix_E[i]-suffix_E[j])
            if 0 <= gamma1 <= 1 and 0 <= gamma2 <= 1:
                return c1, c2, gamma1, gamma2
    for i in range(n):
        c = value[i]
        gamma = (alpha-prefix[i]-suffix[i]) / proba[i]
        if np.isclose(gamma*c*proba[i], alpha*E-prefix_E[i]-suffix_E[i]):
            return c, c, gamma, gamma,
