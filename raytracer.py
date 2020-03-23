
from math import floor, sqrt
from collections import namedtuple
from abc import abstractmethod
from functools import reduce
import sys
import typing

from PIL import Image

from vector import Vector, INF
from color import Color, WHITE, BLACK, GRAY


BASE_MONITORING_RATE = 6400
BASE_BACKUP_RATE = 64000


class Camera(object):
    def __init__(self, pos, look_at):
        self.pos = pos
        down = Vector(0.0, -1.0, 0.0)
        self.forward = (look_at - pos).norm()
        self.right = self.forward.cross(down).norm() * 1.5
        self.up    = self.forward.cross(self.right).norm() * 1.5


Ray          = namedtuple('Ray',          'start dir')
Intersection = namedtuple('Intersection', 'thing ray dist')
Light        = namedtuple('Light',        'pos color')
Surface      = namedtuple('Surface',      'diffuse specular reflect roughness')
Rect         = namedtuple('Rect',         'x y w h')


class Thing(object):
    def __init__(self, surface):
        self.surface = surface

    @abstractmethod
    def intersect(self, ray) -> Intersection:
        pass

    @abstractmethod
    def normal(self, pos) -> Vector:
        pass


class Sphere(Thing):
    def __init__(self, center, radius, surface):
        super(Sphere, self).__init__(surface)
        self.center = center
        self.radius = radius
        self.radius2 = radius ** 2

    def normal(self, pos) -> Vector:
        return (pos - self.center).norm()

    def intersect(self, ray) -> Intersection:
        eo = self.center - ray.start
        v = eo * ray.dir
        dist = 0
        if v >= 0:
            disc = self.radius2 - (eo * eo - v ** 2)
            if disc >= 0:
                dist = v - sqrt(disc)

        if dist != 0:
            return Intersection(self, ray, dist)
        

class Plane(Thing):
    def __init__(self, norm, offset, surface):
        super(Plane, self).__init__(surface)        
        self._normal = norm
        self.offset = offset

    def normal(self, pos) -> Vector:
        return self._normal

    def intersect(self, ray) -> typing.Optional[Intersection]:
        denom = self._normal * ray.dir
        if denom > 0:
            return

        dist = INF if denom == 0 else (-(self._normal * ray.start + self.offset) / denom)
        return Intersection(self, ray, dist)
    

SurfShiny = Surface(
    lambda pos: WHITE,
    lambda pos: GRAY,
    lambda pos: 1.0,
    250
)


SurfCheckerboard = Surface(
    lambda pos: WHITE if (floor(pos.z) + floor(pos.x)) % 2 else BLACK,
    lambda pos: WHITE,
    lambda pos: 0.1 if (floor(pos.z) + floor(pos.x)) % 2 else 0.7,
    150
)


class RayTracer(object):
    maxDepth = 5

    @staticmethod
    def intersections(ray, scene):
        closest, closest_inter = INF, None
        for thing in scene.things:
            inter = thing.intersect(ray)
            if inter is not None and inter.dist < closest:
                closest, closest_inter = inter.dist, inter

        return closest_inter

    def test_ray(self, ray, scene):
        isect = self.intersections(ray, scene)
        return isect and isect.dist

    def trace_ray(self, ray, scene, depth) -> Color:
        isect = self.intersections(ray, scene)
        return isect and self.shade(isect, scene, depth) or scene.background

    def shade(self, isect, scene, depth):
        d = isect.ray.dir
        pos = d * isect.dist + isect.ray.start
        normal = isect.thing.normal(pos)
        reflect_dir = d - (normal * (normal * d)) * 2
        natural_color = scene.background + self.get_natural_color(isect.thing, pos, normal, reflect_dir, scene)
        reflected_color = (
            GRAY
            if depth >= self.maxDepth else
            self.get_reflection_color(isect.thing, pos, normal, reflect_dir, scene, depth)
        )
        return natural_color + reflected_color

    def get_reflection_color(self, thing, pos, normal, rd, scene, depth):
        return thing.surface.reflect(pos) * self.trace_ray(Ray(start=pos, dir=rd), scene, depth + 1)

    def get_natural_color(self, thing, pos, norm, rd, scene):
        def add_light(col, light):
            ldis = light.pos - pos
            livec = ldis.norm()
            neat_isect = self.test_ray(Ray(start=pos, dir=livec), scene)
            if neat_isect is not None and (neat_isect <= abs(ldis)):
                return col
            else:
                illum = livec * norm
                lcolor = (illum * light.color) if illum > 0 else scene.defaultColor
                specular = livec * rd.norm()
                scolor = ((specular ** thing.surface.roughness) * light.color) if specular > 0 else scene.defaultColor
                return col + thing.surface.diffuse(pos) * lcolor + thing.surface.specular(pos) * scolor

        return reduce(add_light, scene.lights, scene.defaultColor)

    def render(self, scene, ctx, screen_width, screen_height, frame=None, state=None, interrupt_rate=None):

        iterator_y = range  # nestedIterAB # range
        iterator_x = range

        kr = screen_height / 4.0 / screen_width
        def get_point(x, y, camera):
            return (
                camera.forward
                + camera.right * (x / 2.0 / screen_width - 0.25)
                + camera.up    * (-y / 2.0 / screen_width + kr)
            ).norm()

        if frame is None:
            frame = Rect(0, 0, screen_width, screen_height)

        percStep = BASE_MONITORING_RATE / frame.w or 1
        percStep = 1 if percStep < 1 else percStep

        for y in iterator_y(max(frame.y, state), frame.y + frame.h):
            if interrupt_rate and y % interrupt_rate == 0:
                yield y

            for x in iterator_x(frame.x, frame.x + frame.w):
                color = self.trace_ray(Ray(start=scene.camera.pos, dir=get_point(x, y, scene.camera)), scene, 0)
                ctx.putpixel((x - frame.x, y - frame.y), color.to_drawing_color())


class Scene(object):
    def __init__(self, things=None, lights=None, camera=None):
        self.things = things or []
        self.lights = lights or []
        self.camera = camera
        self.background = BLACK
        self.defaultColor = BLACK


defScene = Scene(
    things=[
            Sphere(Vector(0.0, 1.0, -0.25), 1.0, SurfShiny),
            Plane(Vector(0.0, 1.0, 0.0), 0.0, SurfCheckerboard),            
            Sphere(Vector(-1.0, 0.5, 1.5), 0.5, SurfShiny)],

    lights=[Light(Vector(-2.0, 2.5,  0.0), Color(0.49, 0.07, 0.07 )),
            Light(Vector( 1.5, 2.5,  1.5), Color(0.07, 0.07, 0.49 )),
            Light(Vector( 1.5, 2.5, -1.5), Color(0.07, 0.49, 0.071)),
            Light(Vector( 0.0, 3.5,  0.0), Color(0.21, 0.21, 0.35))],

    camera=Camera(Vector( 3.0, 2.0,  4.0), Vector(-1.0, 0.5, 0.0))    
)


def state_filename(fn):
    return u'{}.state'.format(fn)


def go(fn, w, h, frame=None):
    import os
    import sys
    import pickle
    from progress_tool import Progress
    
    def backup(w, h, frame, y):
        print('# Save', end='')
        img.save(fn)
        if y < frame.y + frame.h - 1:
            state = dict(w=w, h=h, frame=frame, y=y)
            print(f'Backup render state: {state}', end='')
            with open(state_filename(fn), 'wb') as f:
                pickle.dump(state, f)
        else:
            try:
                print('Try to remove state file...', end='')
                os.remove(state_filename(fn))
                print('ok.')
            except WindowsError:
                print('failed.')

    def resume():
        if os.path.isfile(fn) and os.path.isfile(state_filename(fn)):
            raise IOError('State file not found')
            
        img = Image.open(fn)
        with open(state_filename(fn), 'rb') as f:
            state = pickle.load(f)
            print(f'# Resume render to state: {state!r}')
            return img, state['w'], state['h'], state['frame'], state['y']

    try:
        img, w, h, frame, state = resume()
    except IOError:
        if frame is None:
            frame = Rect(0, 0, w, h)
        img = Image.new('RGB', (frame.w, frame.h))
        state = frame.y

    p = Progress(state, frame.y + frame.h, statusLineStream=sys.stdout)

    echo_rate = BASE_MONITORING_RATE / frame.w or 1
    
    ray_tracer = RayTracer()
    for i, state in enumerate(ray_tracer.render(defScene, img, w, h, frame=frame,
                                               state=state, interrupt_rate=echo_rate)):
        p.next(state)
        if i % 10 == 1:
            backup(w, h, frame, state)

    p.end()
    backup(w, h, frame, state)


def prof():
    import profile
    profile.run('go("render_prof.png", 160, 120)')


def main():
    maxw, maxh = 1920 * 3, 1080 * 3

    if len(sys.argv) == 1:
        # w, h = 1920*6, 1080*6
        w, h = 1024, 768
        frame = Rect(0, 0, w, h)
        # frame = Rect(w/4, h/4, w/2, h/2)
        cx, cy = w / 2, h / 2
    elif len(sys.argv) == 1 + 2:
        w, h = map(eval, sys.argv[1:3])
        frame = Rect(0, 0, w, h)
        cx, cy = w / 2, h / 2
    else:
        k = eval(sys.argv[5])
        ww, hh = map(eval, sys.argv[1:3])
        cx, cy = map(eval, sys.argv[3:5])
        w = ww * k
        h = hh * k
        frame = Rect(
            int(w * cx - ww / 2),
            int(h * cx - hh / 2),
            int(ww), int(hh))

    fn = 'render_{w}x{h}_[({frame.x},{frame.y})-{frame.w}x{frame.h}].png'.format(**locals())
    print(f'''Render to file {fn}
        Size: {w}x{h}
        Frame: ({frame.x},{frame.y}) {frame.w}x{frame.h}
        Center: ({cx}, {cy})
    ''')
    go(fn, w, h, frame=frame)


if __name__ == '__main__':
    main()
