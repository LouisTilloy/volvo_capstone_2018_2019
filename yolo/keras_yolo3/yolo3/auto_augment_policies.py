import numpy as np
from .base_policies import get_transformations

BASE_POLICIES = [t[0] for t in get_transformations()]


def get_aa_policy(i1, p1, m1, i2, p2, m2):
    """
    :param i1: index of the first transformation
    :param p1: probability to apply the first transformation
    :param m1: parameter of the first transformation
    :param i2: index of the second transformation
    :param p2: probability to apply the second transformation
    :param m2: parameter of the second transformation
    :return: policy made from the 2 transformations.
    """
    def policy(img):
        if np.random.rand() < p1:
            img = BASE_POLICIES[i1](img, m1)
        if np.random.rand() < p2:
            img = BASE_POLICIES[i2](img, m2)
        return img
    return policy


def get_aa_policies():
    list_policies = [
        get_aa_policy(11, 0.6, 0.1, 9, 0.5, 4.444),
        get_aa_policy(1, 0.3, -0.1, 10, 0.5, 0.3),
        get_aa_policy(10, 0.1, 0.9, 0, 0.5, 0.3),
        get_aa_policy(1, 0, 0.167, 7, 0.7, 0.333),
        get_aa_policy(10, 0.4, 0.1, 12, 0.2, 0.9),
        get_aa_policy(3, 0.5, -0.15, 12, 0.7, 0.3),
        get_aa_policy(13, 0.8, 1.9, 14, 0.1, 0.178),
        get_aa_policy(15, 0, 0, 14, 0.6, 0.022),
        get_aa_policy(13, 0.9, 1.1, 12, 0.8, 1.1),
        get_aa_policy(11, 0.1, 0.3, 2, 0.7, -0.15),
        get_aa_policy(14, 0.3, 0.067, 6, 0, 0.333),
        get_aa_policy(7, 0, 0.667, 12, 0.2, 1.7),
        get_aa_policy(0, 0.1, 0.033, 12, 0.5, 0.1),
        get_aa_policy(10, 0.1, 0.3, 14, 0.1, 0.067),
        get_aa_policy(8, 0.4, 56.889, 5, 0.5, 0.444),
        get_aa_policy(14, 0.8, 0.133, 7, 0.5, 0.333),
        get_aa_policy(1, 0.2, -0.1, 11, 0.7, 1.1),
        get_aa_policy(12, 0.3, 0.1, 14, 0, 0.178),
        get_aa_policy(14, 0.3, 0, 7, 0, 0.889),
        get_aa_policy(0, 0.5, -0.3, 10, 0.9, 1.3),
    ]
    return list_policies


def get_aa_policies_new():
    list_policies= [
        get_aa_policy(4, 0.4, 10.0, 14, 0.1, 0.067),
        get_aa_policy(7, 0.8, 0.222, 7, 0.8, 1.0),
        get_aa_policy(12, 1.0, 0.3, 10, 0.7, 1.9),
        get_aa_policy(8, 0.6, 227.556, 11, 0.6, 1.3),
        get_aa_policy(3, 0.1, 0.250, 1, 0.7, -0.100),

        get_aa_policy(11, 0.5, 1.1, 10, 0.7, 0.100),
        get_aa_policy(12, 0.6, 0.5, 10, 1.0, 1.7),
        get_aa_policy(9, 0.4, 4.889, 4, 0.7, 2.333),
        get_aa_policy(7, 0.2, 1.0, 10, 0.7, 1.1),
        get_aa_policy(12, 0.5, 1.3, 12, 0.9, 1.7),

        get_aa_policy(12, 0.4, 0.1, 12, 0.6, 1.7),
        get_aa_policy(12, 0.2, 1.5, 1, 0.5, -0.1),
        get_aa_policy(1, 0.6, 0.033, 1, 0.9, 0.033),
        get_aa_policy(5, 0.3, 0.667, 12, 0.2, 0.7),
        get_aa_policy(6, 0.4, 0.111, 1, 0.1, 0.233),

        get_aa_policy(5, 0.9, 0.222, 8, 0.0, 142.222),
        get_aa_policy(5, 0.8, 0.778, 6, 0.9, 0.444),
        get_aa_policy(12, 0.5, 0.1, 10, 1.0, 0.5),
        get_aa_policy(5, 0.9, 0.444, 4, 0.4, 10.0),
        get_aa_policy(8, 0.8, 113.778, 5, 0.7, 0.556)
    ]
    return list_policies
