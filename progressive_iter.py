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


def nestedIterAB(a, b=None):
    if b is None:
        return nestedIterN(a)
    else:
        if a > b:
            a, b = b, a
        return (i + a for i in nestedIterN(b - a))


def nestedIter(l, k=1, s=0):
    n, l = l[0], l[1:]
    for i in range(0, n * k, k):
        if l:
            for j in nestedIter(l, k * n, s + i):
                yield j
        else:
            yield i + s


if __name__ == '__main__':
    print list(nestedIterN(8))

