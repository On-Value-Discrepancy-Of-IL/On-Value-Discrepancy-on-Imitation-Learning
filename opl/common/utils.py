import time


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % (method, args, kw, te-ts))
        # print('%r %2.2f sec' % (method., te - ts))
        return result
    return timed
