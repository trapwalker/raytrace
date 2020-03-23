from typing import NamedTuple
from math import floor


class Color(NamedTuple):
    r: float
    g: float
    b: float

    def scale(self, k):
        return Color(self.r * k, self.g * k, self.b * k)

    def plus(self, c):
        return Color(self.r + c.r, self.g + c.g, self.b + c.b)

    def times(self, c):
        return Color(self.r * c.r, self.g * c.g, self.b * c.b)

    def to_drawing_color(self):
        return Color(int(floor(min(self.r, 1) * 255)),
                     int(floor(min(self.g, 1) * 255)),
                     int(floor(min(self.b, 1) * 255)))

    def __mul__(self, arg):
        return self.times(arg) if isinstance(arg, Color) else self.scale(arg)

    def __str__(self):
        return 'RGB({0.r}, {0.g}, {0.b})'.format(self.to_drawing_color())

    def __repr__(self):
        return 'Color({0.r}, {0.g}, {0.b})'.format(self)

    __add__ = plus
    __rmul__ = __mul__


WHITE = Color(1.0, 1.0, 1.0)
GRAY = Color(0.5, 0.5, 0.5)
BLACK = Color(0.0, 0.0, 0.0)
