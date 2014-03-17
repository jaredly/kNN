#!/usr/bin/env python

from scipy.spatial.distance import cdist
from scipy.io.arff import loadarff
from pandas import DataFrame
from numpy import array
import numpy

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
        dists = cdist(array(data[goods]), array(self.data[goods]), 'eucliedean')
        print time.time() - start
        print 'doneval'

        for i in data.index:
            dlist = dists[i].copy()
            dlist.sort()

            votes = {}
            for ix in range(-self.k, 0):
                cls = self.data.loc[dlist.index[ix]][self.target]
                if not cls in votes:
                    votes[cls] = 1
                else:
                    votes[cls] += 1
            most = None
            for k, v in votes.iteritems():
                if most is None or k > most[0]:
                    most = k, v

            cls = most[1]

            # item = data.loc[i]
            # cls = self.classify(item)
            if data.loc[i][self.target] != cls:
                wrong += 1
            print '.',
        return wrong / float(len(data))

def main(base='mt_'):
    train, mtrain = loadarff(base + 'train.arff')
    train = DataFrame(train)
    test, mtest = loadarff(base + 'test.arff')
    test = DataFrame(test)

    learner = NearestNeighbor(mtrain, train, mtrain.names()[-1])
    import time
    print 'testing'
    start = time.time()
    print learner.validate(test)
    print time.time() - start
    

    

if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
