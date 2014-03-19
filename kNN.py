#!/usr/bin/env python
'''kNN Awesomez

Usage:
    kNN.py [--normalize] [--distance] [--k=<k>] [--base=<base>]
    kNN.py (-h | --help)
    kNN.py --version

Options:
    -h --help       Show the help
    --version       Show the version
    --normalize     Normalize the data
    --distance      Weight the values by the distance
    --k=<n>         the number of neighbors [default: 3]
    --base=<base>   the file base [default: mt_]

'''

from scipy.spatial.distance import cdist
from scipy.io.arff import loadarff
from pandas import DataFrame
from numpy import array
import numpy
import docopt

DEBUG = False

def reuse_recycle(target, data, number, distance=False):
    '''recycle'''
    goods = list(data.columns)
    goods.remove(target)
    dists = DataFrame(cdist(array(data[goods]), array(data[goods]), 'euclidean'))
    remove = []
    for i in dists.index:
        dlist = dists[i].copy()
        dlist.sort()

        votes = {}
        if DEBUG:print dlist.index
        if DEBUG:print dlist

        look = number
        ix = 0
        while look > 0:
            ix -= 1
            if dlist.index[ix] in remove:
                continue
            cls = data.iloc[dlist.index[ix]][target]
            if DEBUG:print ix, dlist.index[ix], cls
            if not cls in votes:
                votes[cls] = (1/dlist.loc[dlist.index[ix]]**2) if distance else 1
            else:
                votes[cls] += (1/dlist.loc[dlist.index[ix]]**2) if distance else 1
            look -= 1
        most = None
        for k, v in votes.iteritems():
            if most is None or v > most[1]:
                most = k, v

        cls = most[0]
        if DEBUG:print votes

        # item = data.loc[i]
        # cls = self.classify(item)
        if data.loc[i][target] == cls:
            remove.append(i)

    print 'removed', len(remove)
    return remove


def reduce_regress(target, data, k, distance=False, maxerr=.1):
    '''recycle'''
    goods = list(data.columns)
    goods.remove(target)
    dists = DataFrame(cdist(array(data[goods]), array(data[goods]), 'euclidean'))
    remove = []
    for i in dists.index:
        dlist = dists[i].copy()
        dlist.sort()

        if distance:
            weighted = 0
            weights = 0
            if DEBUG:print dlist.index
            if DEBUG:print dlist
            look = k
            ix = 0
            while look > 0:
                ix -= 1
                if dlist.index[ix] in remove:
                    continue
                val = data.loc[dlist.index[ix]][target]
                d2i = 1/dlist.loc[dlist.index[ix]]**2
                weighted += val * d2i
                weights += d2i
                look -= 1

            should = weighted / weights

        else:
            total = 0
            look = k
            ix = 0
            while look > 0:
                ix -= 1
                if dlist.index[ix] in remove:
                    continue
                total += data.loc[dlist.index[ix]][target]
                look -= 1

            should = total / ninstances

        err = (data.loc[i][target] - should)**2
        if err < maxerr:
            remove.append(i)

    print 'removed', len(remove)
    return remove

class NearestNeighbor:
    def __init__(self, meta, data, target, k=3, distance=False):
        self.data = data
        self.meta = meta
        self.target = target
        self.distance = distance

    def calc(self, data):
        goods = list(self.data.columns)
        goods.remove(self.target)
        print 'valid'
        import time
        start = time.time()

        # dists is a len(data) x len(self.data) matrix
        dists = DataFrame(cdist(array(self.data[goods]), array(data[goods]), 'euclidean'))
        print time.time() - start
        print 'doneval'
        if DEBUG:print dists
        self.dists = dists

    def regress(self, data, ninstances):
        '''Returns accuracy?'''

        sse = 0 # sum squared error

        for i in data.index:
            dlist = self.dists[i].copy()
            dlist.sort()

            if self.distance:
                weighted = 0
                weights = 0
                if DEBUG:print dlist.index
                if DEBUG:print dlist
                for ix in range(0, ninstances):
                    val = self.data.iloc[dlist.index[ix]][self.target]
                    d2i = 1/dlist.loc[dlist.index[ix]]**2
                    weighted += val * d2i
                    weights += d2i

                should = weighted / weights

            else:
                total = 0
                for ix in range(0, ninstances):
                    total += self.data.loc[dlist.index[ix]][self.target]

                should = total / ninstances

            sse += (data.loc[i][self.target] - should)**2

        print sse, len(data)
        return sse / float(len(data))


    def validate(self, data, ninstances):
        '''Returns accuracy?'''
        wrong = 0

        for i in data.index:
            dlist = self.dists[i].copy()
            dlist.sort()

            votes = {}
            if DEBUG:print dlist.index
            if DEBUG:print dlist
            for ix in range(0, ninstances):
                cls = self.data.iloc[dlist.index[ix]][self.target]
                if DEBUG:print ix, dlist.index[ix], cls
                if not cls in votes:
                    votes[cls] = (1/dlist.loc[dlist.index[ix]]**2) if self.distance else 1
                else:
                    votes[cls] += (1/dlist.loc[dlist.index[ix]]**2) if self.distance else 1
            most = None
            for k, v in votes.iteritems():
                if most is None or v > most[1]:
                    most = k, v

            cls = most[0]
            if DEBUG:print votes

            # item = data.loc[i]
            # cls = self.classify(item)
            if data.loc[i][self.target] != cls:
                wrong += 1
                # print cls, data.loc[i][self.target]
            # print '.',
            # if i > 20:break
        # print wrong, len(data)
        return wrong / float(len(data))

def norms(one, two, cols):
    for c in cols:
        mn = min([one[c].min(), two[c].min()])
        mx = max([one[c].max(), two[c].max()])
        one[c] -= mn
        one[c] /= mx - mn
        two[c] -= mn
        two[c] /= mx - mn

def main(k=3, normalize=False, distance=True, base='mt_', ks=[], regress=False, recycle=False, maxerr=.1):
    train, mtrain = loadarff(base + 'train.arff')
    train = DataFrame(train)
    test, mtest = loadarff(base + 'test.arff')
    test = DataFrame(test)

    cols = [col for col in mtrain.names() if mtrain[col][0] == 'numeric']

    if normalize:
        norms(test, train, cols)

    target = mtrain.names()[-1]
    if recycle:
        print len(train)
        if regress:
            removed = reduce_regress(target, train, k, True, maxerr=maxerr)
        else:
            removed = reuse_recycle(target, train, k, True)
        # print removed
        ixs = list(train.index)
        for n in removed:
            ixs.remove(n)
        train = train.loc[ixs]
        print len(train)
        # print train.index

    learner = NearestNeighbor(mtrain, train, target, distance=distance)
    learner.calc(test)

    tester = learner.regress if regress else learner.validate

    import time
    print 'testing', [k]
    start = time.time()
    err = tester(test, k)
    print 'Err:', err, 'Acc:', 1-err
    print 'Time', time.time() - start
    if not ks: return err
    errs = {}
    errs[k] = err
    for ok in ks:
        print 'testing', ok
        start = time.time()
        err = tester(test, ok)
        print 'Err:', err, 'Acc:', 1-err
        print 'Time', time.time() - start
        errs[ok] = err
    return errs

if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='kNN 1.0')
    main(k=int(args['--k']), distance=args['--distance'], normalize=args['--normalize'], base=args['--base'])

# vim: et sw=4 sts=4
