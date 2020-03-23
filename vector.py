
from math import sqrt
from typing import NamedTuple


INF = float('inf')


class Vector(NamedTuple):
    x: float
    y: float
    z: float

    def times(self, k):
        return Vector(self.x * k, self.y * k, self.z * k)

    def minus(self, v):
        return Vector(self.x - v.x, self.y - v.y, self.z - v.z)

    def plus(self, v):
        return Vector(self.x + v.x, self.y + v.y, self.z + v.z)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def mag(self):
        return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def norm(self):
        mag = self.mag()
        return self.times(INF if mag == 0 else 1.0 / mag)

    def cross(self, v):
        return Vector(self.y * v.z - self.z * v.y,
                      self.z * v.x - self.x * v.z,
                      self.x * v.y - self.y * v.x)

    def __mul__(self, arg):
        return self.dot(arg) if isinstance(arg, Vector) else self.times(arg)

    def __repr__(self):
        return 'Vector({0.x}, {0.y}, {0.z})'.format(self)

    __add__ = plus
    __sub__ = minus
    __abs__ = mag
    __rmul__ = __mul__
