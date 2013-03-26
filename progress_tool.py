# -*- coding: UTF-8 -*-
import sys
import datetime

SECOND = 1 / 24 / 60 / 60

def measureTime(func, out=sys.stdout):
    def cover(*argv, **kw):
        time_start = datetime.datetime.now()
        res = func(*argv, **kw)
        print >> out, 'Ok %.50s' % (datetime.datetime.now() - time_start)
        return res
    
    return cover


class Progress(object):

    def __init__(self, a, b=None, run=True):
        if b is None: a, b = 0, a - 1
        if b < a:     a, b = b, a

        self.a = a
        self.b = b
        self.value = a
        self.isRun = run
        self.tmStart = datetime.datetime.now()
        self.tmPauseBegin = None

    def pause(self):
        if self.isRun:
            self.tmPauseBegin = datetime.datetime.now()
            self.isRun = False

    def run(self):
        if not self.isRun:
            deltaPause = datetime.datetime.now() - self.tmPauseBegin
            self.tmStart += deltaPause
            self.isRun = True

    @property
    def isFinished(self):
        return self.value >= self.b

    def next(self, value=None):
        if not self.isRun:
            self.run()
            
        if value is None:
            self.value += 1
        else:
            self.value = value

        return self.fmt()

    def getState(self):
        size = self.b - self.a
        k = (float(self.value) / size) if size else 1
        perc = k * 100
        time = datetime.datetime.now() - self.tmStart
        remaining = time.total_seconds() * (1 - k) / k if k else 0        
        remaining = datetime.timedelta(seconds=int(remaining), microseconds=int((remaining % 1) * 1000000))
        total = time + remaining

        remaining_str = str(remaining)[:7]
        time_str = str(time)[:7]
        total_str = str(total)[:7]

        d = locals().copy()
        d.update(self.__dict__)
        return d

    def fmt(self, fmt=u'%(perc)6.2f%%\t прошло %(time_str)s из %(total_str)s; осталось %(remaining_str)s'):
        return fmt % self.getState()

def test():
    v = 100000000
    pb = Progress(v)
    for i in xrange(v):
        if i % 5000000 == 0:
            print pb.next(i)

    print pb.next(i)
    
    globals().update(locals())
        

if __name__ == '__main__':
    test()
