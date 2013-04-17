
from math import floor, sqrt
from operator import itemgetter
from collections import namedtuple
from abc import abstractmethod

from PIL import Image

INF = float('inf')
BASE_MONITORING_RATE = 6400
BASE_BACKUP_RATE = 64000

def progressiveRange(n):
    pass


class Vector(tuple):
    x = property(itemgetter(0))
    y = property(itemgetter(1))
    z = property(itemgetter(2))
    
    def __new__(cls, x, y, z):
        return tuple.__new__(cls, (x, y, z))

    def times(self, k): return Vector(self.x * k, self.y * k, self.z * k)
    def minus(self, v): return Vector(self.x - v.x, self.y - v.y, self.z - v.z)
    def plus(self, v):  return Vector(self.x + v.x, self.y + v.y, self.z + v.z)
    def dot(self, v):   return self.x * v.x + self.y * v.y + self.z * v.z
    def mag(self):      return sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    def norm(self):
        mag = self.mag()
        return self.times(INF if mag == 0 else 1.0 / mag)

    def cross(self, v):
        return Vector(self.y * v.z - self.z * v.y,
                      self.z * v.x - self.x * v.z,
                      self.x * v.y - self.y * v.x);

    def __mul__(self, arg):
        return self.dot(arg) if isinstance(arg, Vector) else self.times(arg)

    def __repr__(self):
        return 'Vector({0.x}, {0.y}, {0.z})'.format(self)

    __add__ = plus
    __sub__ = minus
    __abs__ = mag
    __rmul__ = __mul__


class Color(tuple):
    r = property(itemgetter(0))
    g = property(itemgetter(1))
    b = property(itemgetter(2))

    def __new__(cls, r, g, b):
        return tuple.__new__(cls, (r, g, b))

    def scale(self, k): return Color(self.r * k, self.g * k, self.b * k)
    def plus(self, c):  return Color(self.r + c.r, self.g + c.g, self.b + c.b)
    def times(self, c): return Color(self.r * c.r, self.g * c.g, self.b * c.b)

    def toDrawingColor(self):
        return Color(int(floor(min(self.r, 1) * 255)),
                     int(floor(min(self.g, 1) * 255)),
                     int(floor(min(self.b, 1) * 255)))

    def __mul__(self, arg):
        return self.times(arg) if isinstance(arg, Color) else self.scale(arg)

    def __str__(self):
        return 'RGB({0.r}, {0.g}, {0.b})'.format(self.toDrawingColor())
    
    def __repr__(self):
        return 'Color({0.r}, {0.g}, {0.b})'.format(self)

    __add__ = plus
    __rmul__ = __mul__

Color.white = Color(1.0, 1.0, 1.0)
Color.grey  = Color(0.5, 0.5, 0.5);
Color.black = Color(0.0, 0.0, 0.0);
Color.background = Color.black;
Color.defaultColor = Color.black;


class Camera(object):
    def __init__(self, pos, lookAt):
        self.pos = pos
        down = Vector(0.0, -1.0, 0.0)
        self.forward = (lookAt - pos).norm()
        self.right = self.forward.cross(down).norm() * 1.5
        self.up    = self.forward.cross(self.right).norm() * 1.5


Ray          = namedtuple('Ray',          'start dir')
Intersection = namedtuple('Intersection', 'thing ray dist')
Light        = namedtuple('Light',        'pos color')
Surface      = namedtuple('Surface',      'diffuse specular reflect roughness')


class Thing(object):
    def __init__(self, surface):
        self.surface = surface

    @abstractmethod
    def intersect(self, ray): # Intersection
        pass

    @abstractmethod
    def normal(self, pos): # Vector
        pass


class Sphere(Thing):
    def __init__(self, center, radius, surface):
        super(Sphere, self).__init__(surface)
        self.center = center
        self.radius = radius
        self.radius2 = radius ** 2

    def normal(self, pos): # Vector
        return (pos - self.center).norm()

    def intersect(self, ray): # Intersection
        eo = self.center - ray.start
        v = eo * ray.dir
        dist = 0
        if v >= 0:
            #disc = self.radius2 - (Vector.dot(eo, eo) - v * v)
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

    def normal(self, pos): # Vector
        return self._normal

    def intersect(self, ray): # Intersection
        denom = self._normal * ray.dir
        if denom > 0:
            return

        dist = INF if denom == 0 else (-(self._normal * ray.start + self.offset) / denom)
        return Intersection(self, ray, dist)
    

SurfShiny = Surface(
    lambda pos: Color.white,
    lambda pos: Color.grey,
    lambda pos: 1.0,
    250
)


SurfCheckerboard = Surface(
    lambda pos: Color.white if (floor(pos.z) + floor(pos.x)) % 2 else Color.black,
    lambda pos: Color.white,
    lambda pos: 0.1 if (floor(pos.z) + floor(pos.x)) % 2 else 0.7,
    150
)


class RayTracer(object):
    maxDepth = 5

    def intersections(self, ray, scene):        
        closest, closestInter = INF, None
        for thing in scene.things:
            inter = thing.intersect(ray);
            if inter is not None and inter.dist < closest:
                closest, closestInter = inter.dist, inter

        return closestInter

    def testRay(self, ray, scene):
        isect = self.intersections(ray, scene);
        return isect and isect.dist;

    def traceRay(self, ray, scene, depth): # Color
        isect = self.intersections(ray, scene);
        return isect and self.shade(isect, scene, depth) or Color.background

    def shade(self, isect, scene, depth):
        d = isect.ray.dir
        pos = d * isect.dist + isect.ray.start
        normal = isect.thing.normal(pos)
        reflectDir = d - (normal * (normal * d)) * 2
        naturalColor = Color.background + self.getNaturalColor(isect.thing, pos, normal, reflectDir, scene)
        reflectedColor = Color.grey if depth >= self.maxDepth else self.getReflectionColor(isect.thing, pos, normal, reflectDir, scene, depth)
        return naturalColor + reflectedColor

    def getReflectionColor(self, thing, pos, normal, rd, scene, depth):
        return thing.surface.reflect(pos) * self.traceRay(Ray(start=pos, dir=rd), scene, depth + 1)

    def getNaturalColor(self, thing, pos, norm, rd, scene):
        def addLight(col, light):
            ldis = light.pos - pos
            livec = ldis.norm()
            neatIsect = self.testRay(Ray(start=pos, dir=livec), scene)
            if neatIsect is not None and (neatIsect <= abs(ldis)):
                return col
            else:
                illum = livec * norm;
                lcolor = (illum * light.color) if illum > 0 else Color.defaultColor
                specular = livec * rd.norm()
                scolor = ((specular ** thing.surface.roughness) * light.color) if specular > 0 else Color.defaultColor
                return col + thing.surface.diffuse(pos) * lcolor + thing.surface.specular(pos) * scolor

        return reduce(addLight, scene.lights, Color.defaultColor)

    def render(self, scene, ctx, screenWidth, screenHeight, progressCallback=None, state=0, backupRate=100, backupCallback=None):
        def getPoint(x, y, camera):
            return (
                camera.forward
                + camera.right * ( (x - (screenWidth  / 2.0)) / 2.0 / screenWidth)
                + camera.up    * (-(y - (screenHeight / 2.0)) / 2.0 / screenWidth)
            ).norm()

        y = 0
        percStep = BASE_MONITORING_RATE / screenWidth or 1
        percStep = 1 if percStep < 1 else percStep
        
        for y in xrange(state, screenHeight):
            if progressCallback and y % percStep == 0: progressCallback(y)
                
            if backupCallback and y % backupRate == 0:
                backupCallback(screenWidth, screenHeight, y)
                
            for x in xrange(screenWidth):
                color = self.traceRay(Ray(start=scene.camera.pos, dir=getPoint(x, y, scene.camera)), scene, 0)
                ctx.putpixel((x, y), color.toDrawingColor())
                #ctx.addPixel(color.toDrawingColor())

class Scene(object):
    def __init__(self, things=[], lights=[], camera=None):
        self.things = things
        self.lights = lights
        self.camera = camera


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

def stateFileName(fn):
    return u'{}.state'.format(fn)

def go(fn, w, h):
    import os
    import sys
    import pickle
    from progress_tool import Progress
    
    def backup(w, h, y):
        print '# Save',
        img.save(fn)
        if y < h - 1:
            state = dict(w=w, h=h, y=y)
            print 'Backup render state: {!r}'.format(state),
            with open(stateFileName(fn), 'wb') as f:
                pickle.dump(state, f)
        else:
            try:
                print 'Try to remove state file...',
                os.remove(stateFileName(fn))
                print 'ok.'
            except WindowsError:
                print 'failed.'

    def resume():
        if os.path.isfile(fn) and os.path.isfile(stateFileName(fn)):
            raise IOError('State file not found')
            
        img = Image.open(fn)
        with open(stateFileName(fn), 'rb') as f:
            state = pickle.load(f)
            print '# Resume render to state: {!r}'.format(state)
            return img, state['w'], state['h'], state['y']

    try:
        img, w, h, y = resume()
    except IOError:
        img = Image.new('RGB', (w, h))
        y = 0

    p = Progress(y, h, statusLineStream=sys.stdout)
    p.refreshStatusLine()

    rayTracer = RayTracer()
    rayTracer.render(
        defScene, img, w, h, state=y,
        #progressCallback=p.next, 
        #backupRate=BASE_BACKUP_RATE / w or 1, backupCallback=backup,
    )

    p.next(h)
    backup(w, h, h)

if __name__ == '__main__':
    #w, h = 1920*3, 1080*3
    w, h = 320, 240
    fn = 'render_{0}x{1}.png'.format(w, h)
    go(fn, w, h)

    #for i in progressiveRange(64):
    #    print i
