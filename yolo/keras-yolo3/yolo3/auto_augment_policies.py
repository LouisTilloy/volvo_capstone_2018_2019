import numpy as np
from yolo3.base_policies import get_transformations

BASE_POLICIES = [t[0] for t in get_transformations()]


def get_aa_policy(i1, p1, m1, i2, p2, m2):
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