import multiprocessing
import os
import pickle
import shutil
import subprocess

import math
import matplotlib.pyplot as plt

import neat2
import numpy as np
# import vtk_node_data_modify
import fitness_function
import shape
import utils
import data2txt
import random
import time
from utils import create_folder


#Todo 1.初代末代+结构 2.front

def point_xy(orig_size_xy, density):
    l, w = int(orig_size_xy[0] * density), int(orig_size_xy[1] * density)

    x = np.linspace(0, 1, l)
    y = np.linspace(0, 1, w)
    X = np.zeros((w, l))
    Y = np.zeros((w, l))

    X[:, :] = x
    Y[:, :] = y.reshape(-1, 1)
    input_xy = np.stack((X.flatten(), Y.flatten()), axis=1)
    # normalize the input_xyz
    for i in range(2):
        input_xy[:, i] = utils.normalize(input_xy[:, i], 0, 1)  # [-1, 1]
    input_xy[:, 0]*=3
    return input_xy




def get_load_support(pcd):
    load=[]
    support=[]
    for i,point in enumerate(pcd):
        if point[0] == 3 :
            if point[1]<=-0.6:
                load.append(i)

    for i, point in enumerate(pcd):
        if point[0] == -3 :
                support.append(i)
    return load,support

def split2inside(output, pointcloud):
    pcd = pointcloud
    X = pcd[:, 0]
    Y = pcd[:, 1]
    trian_X = X.reshape(shapey, shapex)
    trian_Y = Y.reshape(shapey, shapex)
    a = output.reshape(shapey, shapex)
    train_p_out = a
    if_inside = train_p_out > threshold
    inside_X = trian_X[if_inside]
    inside_Y = trian_Y[if_inside]
    inside = np.stack((inside_X.flatten(), inside_Y.flatten()), axis=1)
    split_parts = {}
    split_parts["inside"] = inside
    split_parts["all_square"] = a
    load,supp=get_load_support(pcd)
    split_parts['load'] = load
    split_parts['support'] = supp
    return split_parts

def copy_dir(src,des):
    # 确保目标文件夹存在
    shutil.copytree(src, des)


def eval_genome(genome, config):
    net = neat2.nn.FeedForwardNetwork.create(genome, config)
    outputs = []
    fitness=[]
    pointcloud2=pointcloud.copy()
    for point in pointcloud2:
        output = net.activate(point)
        outputs.append(output)
    outputs = np.array(outputs)
    outputs = utils.scale(outputs)
    split = split2inside(outputs, pointcloud2)
    inside = split['inside']
    load = split['load']
    load_change = {key: [0, -100] for key in load}
    support = split['support']
    support_change = {key: [1, 1] for key in support}
    outputs_square = split['all_square']
    Index, X, Y, Cat = shape.find_contour(a=outputs_square, thresh=threshold, pcd=pointcloud2)
    x_values = X.flatten()
    y_values = Y.flatten()
    cat_values = Cat.flatten()
    index_values = Index.flatten()
    index_x_y_cat = np.concatenate(
        (index_values.reshape(-1, 1), x_values.reshape(-1, 1), y_values.reshape(-1, 1), cat_values.reshape(-1, 1)),
        axis=1)
    # Todo 1.输出成data文件到指定的目录 2.运行data文件并将fitness写到文件 3.读取文件设置fitness
    # 1,2  1是outside ，2是inside
    tri_cat = shape.split_tri(Tri, index_x_y_cat)
    f,u=fitness_function.fea_analysis(nodes=pointcloud2, elements=Tri, loads=load_change, supps=support_change,
                                            tags=tri_cat) 
    fitness.append(f)
    fitness.append(inside.shape[0])
    if  inside.shape[0]<=pointcloud2.shape[0]*0.1 or inside.shape[0]>=pointcloud2.shape[0]*0.9:
        return [-1,-1],u
    return fitness,u
    # f1,f2=fitness_function.circle_test_score(inside)
    # return [f1,f2]

# 最后一代结束画图
def plotall(config,thresh,pcd,Tri,shapex,shapey):
    
        len_gen0=len([f for f in os.listdir('./output_gen0')])
        len_final=len([f for f in os.listdir('./output')])
        f1_gen0=[]
        f2_gen0=[]
        for i in range(len_gen0):
            path=f"./output_gen0/g{i}/genome"
            g=shape.getshape(path,config,thresh,pcd,Tri,shapex,shapey)
            f1_gen0.append(g.fitness[0])
            f2_gen0.append(g.fitness[1])
        f1=[]
        f2=[]
        for i in range(len_final):
            path=f"./output/g{i}/genome"
            g=shape.getshape(path,config,thresh,pcd,Tri,shapex,shapey)
            f1.append(g.fitness[0])
            f2.append(g.fitness[1])
        # 画front
        plt.figure(figsize=(6, 6))
        plt.xlabel('f1-1/compliance')
        plt.ylabel('f2-volumn')
        plt.title('pareto front')
        plt.scatter(f1_gen0,f2_gen0,c='r',label="gen0 front")
        plt.scatter(f1,f2,c='b',label="best front")
        plt.legend()
        plt.savefig(f'./pareto_front.png')
        plt.close()
        
def run_experiment(config_path, n_generations=100):
    config = neat2.Config(neat2.DefaultGenome, neat2.DefaultReproduction,config_path)
    p = neat2.Population(config)
    pe = neat2.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    best_genomes = p.run(pe.evaluate, n=n_generations)
    for (k,g) in best_genomes.items():
        print(f"{k}: is{g.fitness}")
    p1=[]
    p2=[]
    for i,g in enumerate((list(best_genomes.items()))):
        path=f"./output/g{i}"
        create_folder(path)
        with open(f'{path}/genome.pkl','wb') as f:
            pickle.dump(g[1],f)
    plotall(config=config,thresh=threshold,pcd=pointcloud,Tri=Tri,shapex=shapex,shapey=shapey)

orig_size_xy = (3, 1)
density = 15
n_generations =1000
threshold = 0.5
pointcloud = point_xy(orig_size_xy, density)

# 要求是square
shapex = orig_size_xy[0]*density
shapey = orig_size_xy[1]*density
Tri = shape.triangulation(shapex, shapey)

if __name__ == '__main__':
    random_seed = 33
    random.seed(random_seed)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'maze_config.ini')
    run_experiment(config_path, n_generations=n_generations)



