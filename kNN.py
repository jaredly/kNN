#!/usr/bin/env python

def dist(one, two, target):
    done = one

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
        for test in self.data:
            neighbors.append((dist(test, item, self.target), test[self.target]))
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



# vim: et sw=4 sts=4
