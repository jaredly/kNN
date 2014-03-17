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

def dist(one, two):
    return numpy.linalg.norm(two - one)

class NearestNeighbor:
    def __init__(self, meta, data, target, k=3, distance=False):
        self.data = data
        self.meta = meta
        self.target = target
        self.distance = distance
        self.k = k

    def run(self):
        return False

    def classify(self, item):
        neighbors = []

        trains = self.data[goods]
        for i in trains.index:
            neighbors.append((dist(trains.loc[i], item[goods]), self.data.loc[i][self.target]))
            print ',',
        neighbors.sort()
        nearest = neighbors[-self.k:]
        items = {}
        for d,x in nearest:
            if x not in items:
                items[x] = 0
            items[x] += 1
        its = items.items()
        its.sort(lambda (a,b),(c,d): d - b)
        return its[0][0]

    def validate(self, data):
        '''Returns accuracy?'''
        wrong = 0
        goods = list(self.data.columns)
        goods.remove(self.target)
        print 'valid'
        import time
        start = time.time()
        # dists is a len(data) x len(self.data) matrix
        dists = DataFrame(cdist(array(data[goods]), array(self.data[goods]), 'euclidean'))
        print time.time() - start
        print 'doneval'

        for i in data.index:
            dlist = dists[i].copy()
            dlist.sort()

            votes = {}
            for ix in range(-self.k, 0):
                cls = self.data.loc[dlist.index[ix]][self.target]
                if not cls in votes:
                    votes[cls] = (1/dlist.loc[ix]) if self.distance else 1
                else:
                    votes[cls] += (1/dlist.loc[ix]) if self.distance else 1
            most = None
            for k, v in votes.iteritems():
                if most is None or v > most[1]:
                    most = k, v

            cls = most[0]
            # print votes

            # item = data.loc[i]
            # cls = self.classify(item)
            if data.loc[i][self.target] != cls:
                wrong += 1
                # print cls, data.loc[i][self.target]
            # print '.',
            # if i > 20:break
        print wrong, len(data)
        return wrong / float(len(data))

def norms(one, two):
    for c in one.columns:
        mn = min([one[c].min(), two[c].min()])
        mx = max([one[c].max(), two[c].max()])
        one[c] -= mn
        one[c] /= mx - mn
        two[c] -= mn
        two[c] /= mx - mn

def main(k=3, normalize=False, distance=True, base='mt_'):
    train, mtrain = loadarff(base + 'train.arff')
    train = DataFrame(train)
    test, mtest = loadarff(base + 'test.arff')
    test = DataFrame(test)

    if normalize:
        norms(test, train)

    learner = NearestNeighbor(mtrain, train, mtrain.names()[-1], k=k, distance=distance)
    import time
    print 'testing'
    start = time.time()
    print learner.validate(test)
    print 'Time', time.time() - start

if __name__ == '__main__':
    args = docopt.docopt(__doc__, version='kNN 1.0')
    main(k=int(args['--k']), distance=args['--distance'], normalize=args['--normalize'], base=args['--base'])

# vim: et sw=4 sts=4
