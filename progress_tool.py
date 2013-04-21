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

    def __init__(self, a, b=None, run=True, statusLineStream=None, statusLineClear=False, 
                 fmt=u'%(perc)6.2f%%\t прошло %(time_str)s из %(total_str)s; осталось %(remaining_str)s'):
        if b is None: a, b = 0, a - 1
        if b < a:     a, b = b, a

        self.a = a
        self.b = b
        self.value = a
        self.isRun = run
        self.tmStart = datetime.datetime.now()
        self.tmPauseBegin = None
        self._statusLineSize = 0
        self.statusLineStream = statusLineStream
        self.statusLineClear = statusLineClear
        self.logFmt = fmt

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

    def end(self):
        self.value = self.b
        self.isRun = False

        if self.statusLineStream:
            return self.refreshStatusLine()
        else:
            return self.fmt()

    def next(self, value=None):
        if not self.isRun:
            self.run()
            
        if value is None:
            self.value += 1
        else:
            self.value = value

        if self.statusLineStream:
            return self.refreshStatusLine()
        else:
            return self.fmt()

    def getState(self):
        size = self.b - self.a
        k = (float(self.value - self.a) / size) if size else 1
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

    def fmt(self, fmt=None):
        return (fmt or self.logFmt) % self.getState()

    def refreshStatusLine(self):
        if self._statusLineSize > 0:
            self.statusLineStream.write('\r' if self.statusLineClear else '\n')

        s = self.fmt()
        self.statusLineStream.write(s)
        if self.statusLineClear:
            self.statusLineStream.write(' ' * (self._statusLineSize - len(s)))
        self._statusLineSize = len(s)
        return s


def test():
    import sys
    b = 100
    a = 0 #b / 2
    pb = Progress(a, b, statusLineStream=sys.stdout)
    for i in xrange(a, b):
        pb.next(i)
        for j in xrange(1000000):
            pass

    print pb.next(i)
    
    globals().update(locals())
        

if __name__ == '__main__':
    test()

