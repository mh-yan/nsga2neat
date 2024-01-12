import pickle
import neat2
import math
import numpy

import main


import shape

config = neat2.Config(
    neat2.DefaultGenome, neat2.DefaultReproduction,
    'maze_config.ini')


orig_size_xy = (3, 1)
density = 15
threshold = 0.5
pointcloud = main.point_xy(orig_size_xy, density)
# 要求是square
shapex = orig_size_xy[0]*density
shapey = orig_size_xy[1]*density
Tri=shape.triangulation(shapex,shapey)
index=7
for i in range(20):
    with open(f'./output/best_genome_{i}.pkl', 'rb') as f:
        g=pickle.load(f)
        print(g.fitness)

shape.getshape(f'./output/best_genome_{0}.pkl', config, threshold, pointcloud, Tri,shapex,shapey)
