from contextlib import contextmanager

import numpy as np
import cv2
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils
from samples.iterative_training.Utils import contexts

class TestTest():
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print('enter', self.name)
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exit', self.name)


def main():
    with contexts(TestTest('111'), TestTest('222')) as (t1, t2):
        print(t1, t2)


main()
