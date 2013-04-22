# -*- coding: UTF-8 -*-

from math import ceil
from itertools import groupby


def simplyfy(n):
    n = abs(n)
    i = 2
    while i <= n:
        while n % i == 0:
            n /= i
            yield i
        i += 1


def nestedIterN(n):
    return nestedIter(list(simplyfy(n)))


def nestedIter(l, k=1, s=0):
    if len(l) > 1:
        for i in range(0, l[0] * k, k):
            for j in nestedIter(l[1:], k * l[0], s + i):
                yield j
    else:
        for i in range(0, l[0] * k, k):
            yield i + s
        

if __name__ == '__main__':
    pass    

