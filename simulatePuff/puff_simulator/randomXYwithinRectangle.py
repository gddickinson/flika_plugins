# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:43:12 2021

@author: g_dic
"""

from matplotlib import pyplot as plt
import random

radius = 200
rangeX = (0, 2500)
rangeY = (0, 2500)
qty = 10  # or however many points you want

# Generate a set of all points within 200 of the origin, to be used as offsets later
# There's probably a more efficient way to do this.
deltas = set()
for x in range(-radius, radius+1):
    for y in range(-radius, radius+1):
        if x*x + y*y <= radius*radius:
            deltas.add((x,y))

randPoints = []
excluded = set()
i = 0
while i<qty:
    x = random.randrange(*rangeX)
    y = random.randrange(*rangeY)
    if (x,y) in excluded: continue
    randPoints.append((x,y))
    i += 1
    excluded.update((x+dx, y+dy) for (dx,dy) in deltas)

print (randPoints)
X = [i[0] for i in randPoints]
Y = [i[1] for i in randPoints]
plt.scatter(X,Y)